using Flux
using Random
using Serialization
using Statistics

const STATE_DIM = 34
const FEATURE_DIM = 32
const ACTION_DIM = 24
const EMBED_DIM = 48
const NUM_HEADS = 4
const HIDDEN_DIM = 96
const TOKEN_COUNT = 3
const CATALOG_VERSION = 2
const REWARD_VERSION = 2
const LEGAL_MASK_ENCODING = "u32-low-high-words-33-34"
const VALID_ACTION_MASK = (UInt64(1) << ACTION_DIM) - UInt64(1)

const ACTION_WAIT = 0
const ACTION_GATHER_GOLD = 1
const ACTION_GATHER_WOOD = 2
const ACTION_BUILD_TOWN_HALL = 3
const ACTION_TRAIN_WORKER = 4
const ACTION_BUILD_FARM = 5
const ACTION_BUILD_BARRACKS = 6
const ACTION_BUILD_LUMBER_MILL = 7
const ACTION_BUILD_BLACKSMITH = 8
const ACTION_BUILD_STABLES = 9
const ACTION_TRAIN_SOLDIER = 10
const ACTION_TRAIN_SHOOTER = 11
const ACTION_TRAIN_CAVALRY = 12
const ACTION_TRAIN_CATAPULT = 13
const ACTION_RESEARCH_WEAPON = 14
const ACTION_RESEARCH_ARMOR = 15
const ACTION_ATTACK_NEAREST_UNIT = 16
const ACTION_ATTACK_NEAREST_BUILDING = 17
const ACTION_ATTACK_WEAKEST_UNIT = 18
const ACTION_ATTACK_WEAKEST_BUILDING = 19
const ACTION_DEFEND_BASE = 20
const ACTION_EXPLORE = 21
const ACTION_PREPARE_BUILDING_SPACE = 22
const ACTION_REPAIR_BUILDING = 23

const ACTION_NAMES = (
 "wait",
 "gather-gold",
 "gather-wood",
 "build-town-hall",
 "train-worker",
 "build-farm",
 "build-barracks",
 "build-lumber-mill",
 "build-blacksmith",
 "build-stables",
 "train-soldier",
 "train-shooter",
 "train-cavalry",
 "train-catapult",
 "research-weapon",
 "research-armor",
 "attack-nearest-unit",
 "attack-nearest-building",
 "attack-weakest-unit",
 "attack-weakest-building",
 "defend-base",
 "explore",
 "prepare-building-space",
 "repair-building",
)

const MODE_INFERENCE = :inference
const MODE_TRAIN = :train
const MODE_RESET_TRAIN = :reset_train
const CHECKPOINT_VERSION = 3
const DEFAULT_BATCH_SIZE = 32
const DEFAULT_CHECKPOINT_EVERY = 32

"""Return the stable name for a zero-based primitive action ID."""
function action_name(action::Integer)::String
 0 <= action < ACTION_DIM || throw(ArgumentError("action $action is outside the AI action range"))
 return ACTION_NAMES[Int(action)+1]
end

"""State encoder, self-attention block, policy head, and scalar value head."""
struct AiPolicy
 state_encoder
 token_encoder
 attention
 feed_forward
 action_head
 value_head
end

Flux.@layer AiPolicy

function create_policy(; seed::Integer=0x574131)
 rng = MersenneTwister(seed)
 init = (dimensions...) -> Flux.glorot_uniform(rng, dimensions...)
 return AiPolicy(
  Dense(FEATURE_DIM => EMBED_DIM, tanh; init),
  Dense(EMBED_DIM => TOKEN_COUNT * EMBED_DIM, tanh; init),
  Flux.MultiHeadAttention(EMBED_DIM; nheads=NUM_HEADS, dropout_prob=0.0f0, init),
  Chain(Dense(EMBED_DIM => HIDDEN_DIM, relu; init), Dense(HIDDEN_DIM => EMBED_DIM; init)),
  Dense(EMBED_DIM => ACTION_DIM; init),
  Dense(EMBED_DIM => 1; init),
 )
end

"""Validate a complete v2 producer state including its nonempty legal-action mask."""
function validate_state(state::AbstractVector{<:Integer})::Nothing
 length(state) == STATE_DIM || throw(ArgumentError("AI state has $(length(state)) fields, expected $STATE_DIM"))
 UInt32(state[1]) == UInt32(2) || throw(ArgumentError("unsupported AI state protocol version $(state[1])"))
 mask = legal_mask(state)
 mask != 0 || throw(ArgumentError("AI state has no legal actions"))
 (mask & ~VALID_ACTION_MASK) == 0 || throw(ArgumentError("AI state has action bits outside the catalog"))
 (mask & UInt64(1)) != 0 || throw(ArgumentError("AI state must allow wait"))
 return nothing
end

"""Decode the producer's low/high UInt32 legality words into a zero-based action mask."""
function legal_mask(state::AbstractVector{<:Integer})::UInt64
 length(state) == STATE_DIM || throw(ArgumentError("AI state has $(length(state)) fields, expected $STATE_DIM"))
 return UInt64(UInt32(state[33])) | (UInt64(UInt32(state[34])) << 32)
end

@inline function is_legal_action(state::AbstractVector{<:Integer}, action::Integer)::Bool
 return 0 <= action < ACTION_DIM && (legal_mask(state) & (UInt64(1) << action)) != 0
end

"""Map the 32 model features, excluding producer-owned legality words, to bounded inputs."""
function encode_state(state::AbstractVector{<:Integer})::Vector{Float32}
 validate_state(state)
 values = Float32.(state[1:FEATURE_DIM])
 scales = Float32[
  2, 7, 1, 20_000, 2_000, 2_000, 100, 100, 50, 10, 10, 10, 10, 10, 50, 50,
  50, 50, 100, 100, 50, 50, 50, 10, 10, 10, 10, 10, 50, 50, 50, 50,
 ]
 return clamp.(values ./ scales, 0.0f0, 4.0f0)
end

function policy_features(policy::AiPolicy, state::AbstractVector{<:Integer})
 encoded = policy.state_encoder(encode_state(state))
 tokens = reshape(policy.token_encoder(encoded), EMBED_DIM, TOKEN_COUNT, 1)
 attended = policy.attention(tokens)[1]
 transformed = tokens .+ attended
 transformed = transformed .+ policy.feed_forward(transformed)
 return vec(mean(transformed; dims=(2, 3)))
end

function action_logits(policy::AiPolicy, state::AbstractVector{<:Integer})::Vector{Float32}
 validate_state(state)
 return Float32.(policy.action_head(policy_features(policy, state)))
end

function value_estimate(policy::AiPolicy, state::AbstractVector{<:Integer})::Float32
 return Float32(only(policy.value_head(policy_features(policy, state))))
end

@inline state_count(state::AbstractVector{<:Integer}, index::Integer) = Int(state[index])

"""Small low-level bootstrap preferences; legality always remains producer-owned."""
function action_bootstrap_biases(state::AbstractVector{<:Integer})::NTuple{ACTION_DIM,Float32}
 validate_state(state)
 gold = state_count(state, 5)
 wood = state_count(state, 6)
 supply = state_count(state, 7)
 demand = state_count(state, 8)
 workers = state_count(state, 9)
 town_halls = state_count(state, 10)
 barracks = state_count(state, 11)
 lumber_mills = state_count(state, 12)
 blacksmiths = state_count(state, 13)
 stables = state_count(state, 14)
 combatants = sum(state_count(state, index) for index in 15:18)
 enemy_units = state_count(state, 21)
 enemy_buildings = state_count(state, 22)

 return ntuple(ACTION_DIM) do index
  action = index - 1
  action == ACTION_GATHER_GOLD && workers > 0 ? (gold < wood ? 0.9f0 : 0.7f0) :
  action == ACTION_GATHER_WOOD && workers > 0 ? (wood < gold ? 0.9f0 : 0.7f0) :
  action == ACTION_BUILD_TOWN_HALL && town_halls == 0 ? 3.0f0 :
  action == ACTION_TRAIN_WORKER && workers < 5 ? 2.0f0 :
  action == ACTION_BUILD_FARM && demand + 2 >= supply ? 2.5f0 :
  action == ACTION_BUILD_BARRACKS && barracks == 0 ? 1.8f0 :
  action == ACTION_BUILD_LUMBER_MILL && lumber_mills == 0 ? 1.7f0 :
  action == ACTION_BUILD_BLACKSMITH && blacksmiths == 0 ? 1.6f0 :
  action == ACTION_BUILD_STABLES && stables == 0 ? 1.5f0 :
  action == ACTION_TRAIN_SOLDIER && barracks > 0 && combatants < 8 ? 1.0f0 :
  action == ACTION_TRAIN_SHOOTER && lumber_mills > 0 && combatants < 8 ? 0.9f0 :
  action == ACTION_TRAIN_CAVALRY && stables > 0 ? 0.8f0 :
  action == ACTION_TRAIN_CATAPULT && blacksmiths > 0 ? 0.8f0 :
  action == ACTION_PREPARE_BUILDING_SPACE &&
   (barracks == 0 || lumber_mills == 0 || blacksmiths == 0 || stables == 0) ? 2.2f0 :
  action == ACTION_ATTACK_NEAREST_UNIT && combatants > 0 && enemy_units > 0 ? 1.4f0 :
  action == ACTION_ATTACK_NEAREST_BUILDING && combatants > 0 && enemy_buildings > 0 ? 1.3f0 :
  0.0f0
 end
end

"""Mask scores with producer authority. Masked actions are never selectable."""
function mask_action_scores(scores::AbstractVector{<:AbstractFloat}, state::AbstractVector{<:Integer})::Vector{Float32}
 length(scores) == ACTION_DIM || throw(ArgumentError("AI action scores have $(length(scores)) entries, expected $ACTION_DIM"))
 validate_state(state)
 legality = ntuple(index -> is_legal_action(state, index - 1), ACTION_DIM)
 return Float32.(ifelse.(legality, scores, -Inf32))
end

function action_scores(policy::AiPolicy, state::AbstractVector{<:Integer})::Vector{Float32}
 return mask_action_scores(action_logits(policy, state) .+ action_bootstrap_biases(state), state)
end

function legal_action_distribution(policy::AiPolicy, state::AbstractVector{<:Integer})
 scores = action_scores(policy, state)
 legal = findall(isfinite, scores)
 isempty(legal) && throw(ArgumentError("AI state has no legal actions"))
 return legal, Flux.softmax(scores[legal])
end

function sample_legal_action(scores::AbstractVector{<:AbstractFloat}, rng::AbstractRNG)::Int
 legal = findall(isfinite, scores)
 isempty(legal) && throw(ArgumentError("AI state has no legal actions"))
 probabilities = Flux.softmax(scores[legal])
 threshold = rand(rng)
 cumulative = zero(eltype(probabilities))
 for (offset, probability) in enumerate(probabilities)
  cumulative += probability
  threshold <= cumulative && return legal[offset] - 1
 end
 return legal[end] - 1
end

"""Choose greedily for inference, or sample only producer-legal actions during training."""
function select_action(
 policy::AiPolicy,
 state::AbstractVector{<:Integer};
 training::Bool=false,
 rng::AbstractRNG=Random.default_rng(),
)::Int
 scores = action_scores(policy, state)
 return training ? sample_legal_action(scores, rng) : argmax(scores) - 1
end

function default_checkpoint_path()::String
 state_home = get(ENV, "XDG_STATE_HOME", "")
 root = isempty(state_home) ? joinpath(homedir(), ".local", "state") : state_home
 return joinpath(root, "war1gus", "actor_critic.jls")
end

function checkpoint_payload(policy::AiPolicy, optimizer_state, update_count::Integer)
 return (
  version=CHECKPOINT_VERSION,
  state_dim=STATE_DIM,
  feature_dim=FEATURE_DIM,
  action_dim=ACTION_DIM,
  catalog_version=CATALOG_VERSION,
  reward_version=REWARD_VERSION,
  mask_encoding=LEGAL_MASK_ENCODING,
  update_count=Int(update_count),
  model_state=Flux.state(policy),
  optimizer_state=optimizer_state,
 )
end

function save_policy_checkpoint!(
 path::AbstractString,
 policy::AiPolicy,
 optimizer_state,
 update_count::Integer,
)::Nothing
 update_count >= 0 || throw(ArgumentError("checkpoint update count must not be negative"))
 directory = dirname(path)
 mkpath(directory)
 temporary = joinpath(directory, ".$(basename(path)).$(getpid()).$(time_ns()).tmp")
 try
  open(temporary, "w") do io
   serialize(io, checkpoint_payload(policy, optimizer_state, update_count))
   flush(io)
  end
  mv(temporary, path; force=true)
 catch
  ispath(temporary) && rm(temporary; force=true)
  rethrow()
 end
 return nothing
end

function load_policy_checkpoint!(path::AbstractString, policy::AiPolicy, fresh_optimizer_state)
 isfile(path) || return (update_count=0, optimizer_state=fresh_optimizer_state)
 payload = open(deserialize, path)
 payload isa NamedTuple || throw(ArgumentError("AI checkpoint has an invalid format"))
 required = (
  :version,
  :state_dim,
  :feature_dim,
  :action_dim,
  :catalog_version,
  :reward_version,
  :mask_encoding,
  :update_count,
  :model_state,
  :optimizer_state,
 )
 all(field -> hasproperty(payload, field), required) ||
  throw(ArgumentError("AI checkpoint is missing v3 compatibility metadata"))
 payload.version == CHECKPOINT_VERSION ||
  throw(ArgumentError("AI checkpoint version $(payload.version) is unsupported"))
 Int(payload.state_dim) == STATE_DIM ||
  throw(ArgumentError("AI checkpoint state dimension $(payload.state_dim) does not match $STATE_DIM"))
 Int(payload.feature_dim) == FEATURE_DIM ||
  throw(ArgumentError("AI checkpoint feature dimension $(payload.feature_dim) does not match $FEATURE_DIM"))
 Int(payload.action_dim) == ACTION_DIM ||
  throw(ArgumentError("AI checkpoint action dimension $(payload.action_dim) does not match $ACTION_DIM"))
 Int(payload.catalog_version) == CATALOG_VERSION ||
  throw(ArgumentError("AI checkpoint catalog version $(payload.catalog_version) is unsupported"))
 Int(payload.reward_version) == REWARD_VERSION ||
  throw(ArgumentError("AI checkpoint reward version $(payload.reward_version) is unsupported"))
 payload.mask_encoding == LEGAL_MASK_ENCODING ||
  throw(ArgumentError("AI checkpoint mask encoding $(repr(payload.mask_encoding)) is unsupported"))
 payload.update_count isa Integer && payload.update_count >= 0 ||
  throw(ArgumentError("AI checkpoint has an invalid update count"))
 typeof(payload.optimizer_state) == typeof(fresh_optimizer_state) ||
  throw(ArgumentError("AI checkpoint optimizer state is incompatible"))
 try
  Flux.loadmodel!(policy, payload.model_state)
 catch error
  throw(ArgumentError("AI checkpoint model parameters are incompatible: $(sprint(showerror, error))"))
 end
 return (update_count=Int(payload.update_count), optimizer_state=payload.optimizer_state)
end

function reset_policy_checkpoint!(path::AbstractString)::Nothing
 ispath(path) && rm(path; force=true)
 return nothing
end

struct Transition
 state::Vector{UInt32}
 action::Int
 legal_mask::UInt64
 reward::Int32
 next_state::Union{Nothing,Vector{UInt32}}
 next_legal_mask::Union{Nothing,UInt64}
end

function Transition(
 state::Vector{UInt32},
 action::Integer,
 reward::Int32,
 next_state::Union{Nothing,Vector{UInt32}},
)
 validate_state(state)
 0 <= action < ACTION_DIM || throw(ArgumentError("action $action is outside the AI action range"))
 is_legal_action(state, action) || throw(ArgumentError("action $action is not legal for this AI state"))
 !isnothing(next_state) && validate_state(next_state)
 return Transition(
  state,
  Int(action),
  legal_mask(state),
  reward,
  next_state,
  isnothing(next_state) ? nothing : legal_mask(next_state),
 )
end

mutable struct OnlineTrainer{O,R<:AbstractRNG}
 policy::AiPolicy
 optimizer_state::O
 lock::ReentrantLock
 mode::Symbol
 gamma::Float32
 entropy_coefficient::Float32
 update_count::Int
 checkpoint_path::String
 batch_size::Int
 checkpoint_every::Int
 pending_transitions::Vector{Transition}
 rng::R
end

is_training(trainer::OnlineTrainer)::Bool = trainer.mode == MODE_TRAIN || trainer.mode == MODE_RESET_TRAIN

function create_trainer(
 ;
 mode::Symbol=MODE_INFERENCE,
 checkpoint_path::AbstractString=default_checkpoint_path(),
 seed::Integer=0x574131,
 gamma::Real=0.99f0,
 entropy_coefficient::Real=0.01f0,
 batch_size::Integer=DEFAULT_BATCH_SIZE,
 checkpoint_every::Integer=DEFAULT_CHECKPOINT_EVERY,
 policy::Union{Nothing,AiPolicy}=nothing,
)
 mode in (MODE_INFERENCE, MODE_TRAIN, MODE_RESET_TRAIN) || throw(ArgumentError("unknown AI mode: $mode"))
 batch_size > 0 || throw(ArgumentError("batch size must be positive"))
 checkpoint_every > 0 || throw(ArgumentError("checkpoint interval must be positive"))
 0.0 <= gamma <= 1.0 || throw(ArgumentError("discount factor must be in [0, 1]"))
 entropy_coefficient >= 0 || throw(ArgumentError("entropy coefficient must not be negative"))

 selected_policy = isnothing(policy) ? create_policy(seed=seed) : policy
 selected_path = String(checkpoint_path)
 optimizer = Flux.OptimiserChain(Flux.ClipNorm(1.0f0), Flux.Adam(1.0f-3))
 fresh_optimizer_state = Flux.setup(optimizer, selected_policy)
 restored = if mode == MODE_RESET_TRAIN
  reset_policy_checkpoint!(selected_path)
  (update_count=0, optimizer_state=fresh_optimizer_state)
 else
  load_policy_checkpoint!(selected_path, selected_policy, fresh_optimizer_state)
 end
 return OnlineTrainer(
  selected_policy,
  restored.optimizer_state,
  ReentrantLock(),
  mode,
  Float32(gamma),
  Float32(entropy_coefficient),
  restored.update_count,
  selected_path,
  Int(batch_size),
  Int(checkpoint_every),
  Transition[],
  MersenneTwister(seed),
 )
end

function save_checkpoint!(trainer::OnlineTrainer)::Nothing
 lock(trainer.lock) do
  save_policy_checkpoint!(
   trainer.checkpoint_path,
   trainer.policy,
   trainer.optimizer_state,
   trainer.update_count,
  )
 end
 return nothing
end

function action_log_probability_and_entropy(policy::AiPolicy, state::AbstractVector{<:Integer}, action::Integer)
 legal, probabilities = legal_action_distribution(policy, state)
 position = findfirst(==(Int(action) + 1), legal)
 isnothing(position) && throw(ArgumentError("action $action is not legal for this AI state"))
 return log(probabilities[position]), -sum(probabilities .* log.(probabilities))
end

function _batch_targets_and_advantages(trainer::OnlineTrainer, transitions::Vector{Transition})
 targets = Vector{Float32}(undef, length(transitions))
 advantages = Vector{Float32}(undef, length(transitions))
 for index in eachindex(transitions)
  transition = transitions[index]
  value = value_estimate(trainer.policy, transition.state)
  bootstrap = isnothing(transition.next_state) ? 0.0f0 : value_estimate(trainer.policy, transition.next_state)
  target = Float32(transition.reward) +
           (isnothing(transition.next_state) ? 0.0f0 : trainer.gamma * bootstrap)
  targets[index] = target
  advantages[index] = target - value
 end
 return targets, advantages
end

function _batch_actor_critic_loss(
 policy::AiPolicy,
 transitions::Vector{Transition},
 targets::Vector{Float32},
 advantages::Vector{Float32},
 entropy_coefficient::Float32,
)
 total_loss = 0.0f0
 for index in eachindex(transitions)
  transition = transitions[index]
  value = value_estimate(policy, transition.state)
  log_probability, entropy = action_log_probability_and_entropy(policy, transition.state, transition.action)
  actor_loss = -advantages[index] * log_probability
  critic_loss = 0.5f0 * (value - targets[index])^2
  total_loss += actor_loss + critic_loss - entropy_coefficient * entropy
 end
 return total_loss / Float32(length(transitions))
end

"""Drain queued transitions in arrival order and perform exactly one shared update."""
function _drain_pending_locked!(trainer::OnlineTrainer)::Union{Nothing,Float32}
 isempty(trainer.pending_transitions) && return nothing
 transitions = trainer.pending_transitions
 targets, advantages = _batch_targets_and_advantages(trainer, transitions)
 result = Flux.withgradient(trainer.policy) do policy
  _batch_actor_critic_loss(policy, transitions, targets, advantages, trainer.entropy_coefficient)
 end
 loss = Float32(result.val)
 isfinite(loss) || throw(ArgumentError("actor-critic update produced a non-finite loss"))
 Flux.update!(trainer.optimizer_state, trainer.policy, result.grad[1])
 trainer.pending_transitions = Transition[]
 trainer.update_count += 1
 trainer.update_count % trainer.checkpoint_every == 0 &&
  save_policy_checkpoint!(
   trainer.checkpoint_path,
   trainer.policy,
   trainer.optimizer_state,
   trainer.update_count,
  )
 return loss
end

function _enqueue_transition_locked!(
 trainer::OnlineTrainer,
 state::Vector{UInt32},
 action::Integer,
 reward::Int32,
 next_state::Union{Nothing,Vector{UInt32}},
)::Union{Nothing,Float32}
 is_training(trainer) || return nothing
 0 <= action < ACTION_DIM || throw(ArgumentError("action $action is outside the AI action range"))
 push!(trainer.pending_transitions, Transition(state, Int(action), reward, next_state))
 return length(trainer.pending_transitions) >= trainer.batch_size ? _drain_pending_locked!(trainer) : nothing
end

"""Queue a non-terminal TD transition; a full queue performs one batch update."""
function enqueue_transition!(
 trainer::OnlineTrainer,
 state::Vector{UInt32},
 action::Integer,
 reward::Int32,
 next_state::Vector{UInt32},
)::Union{Nothing,Float32}
 lock(trainer.lock) do
  return _enqueue_transition_locked!(trainer, state, action, reward, next_state)
 end
end

function enqueue_transition!(
 trainer::OnlineTrainer,
 state::Vector{UInt32},
 action::Integer,
 reward::Integer,
 next_state::Vector{UInt32},
)::Union{Nothing,Float32}
 typemin(Int32) <= reward <= typemax(Int32) || throw(ArgumentError("reward is outside Int32 range"))
 return enqueue_transition!(trainer, state, action, Int32(reward), next_state)
end

update_transition!(args...) = enqueue_transition!(args...)

"""Flush all queued transitions. Terminal frames use this after adding their own transition."""
function flush_transitions!(trainer::OnlineTrainer)::Union{Nothing,Float32}
 lock(trainer.lock) do
  return is_training(trainer) ? _drain_pending_locked!(trainer) : nothing
 end
end

"""Queue a terminal transition and flush its non-empty partial batch without bootstrap."""
function update_terminal!(
 trainer::OnlineTrainer,
 state::Vector{UInt32},
 action::Integer,
 reward::Int32,
)::Union{Nothing,Float32}
 lock(trainer.lock) do
  is_training(trainer) || return nothing
  _enqueue_transition_locked!(trainer, state, action, reward, nothing)
  return _drain_pending_locked!(trainer)
 end
end

function update_terminal!(
 trainer::OnlineTrainer,
 state::Vector{UInt32},
 action::Integer,
 reward::Integer,
)::Union{Nothing,Float32}
 typemin(Int32) <= reward <= typemax(Int32) || throw(ArgumentError("reward is outside Int32 range"))
 return update_terminal!(trainer, state, action, Int32(reward))
end

mutable struct ClientSession
 previous_state::Union{Nothing,Vector{UInt32}}
 previous_action::Union{Nothing,Int}
 last_sequence::Union{Nothing,UInt32}
end

ClientSession() = ClientSession(nothing, nothing, nothing)

"""Credit the previous client action, then select and record the current action."""
function process_step!(
 trainer::OnlineTrainer,
 session::ClientSession,
 sequence::UInt32,
 reward::Int32,
 state::Vector{UInt32},
)::Int
 lock(trainer.lock) do
  if !isnothing(session.last_sequence)
   if sequence == session.last_sequence
    return session.previous_action::Int
   end
   sequence == session.last_sequence + one(UInt32) ||
    throw(ArgumentError("out-of-order AI step sequence $sequence"))
  end
  previous_state = session.previous_state
  if is_training(trainer) && !isnothing(previous_state)
   previous_action = session.previous_action::Int
   _enqueue_transition_locked!(trainer, previous_state, previous_action, reward, state)
   log_event(
    "training_sample";
    session_id=previous_state[2],
    sequence,
    state=previous_state,
    legal_mask=legal_mask(previous_state),
    action=previous_action,
    action_name=action_name(previous_action),
    reward,
    next_state=state,
    next_legal_mask=legal_mask(state),
    terminal=false,
   )
  end
  action = select_action(trainer.policy, state; training=is_training(trainer), rng=trainer.rng)
  session.previous_state = state
  session.previous_action = action
  session.last_sequence = sequence
  return action
 end
end

"""Credit the final client action, flush a partial batch, and persist training."""
function process_terminal!(
 trainer::OnlineTrainer,
 session::ClientSession,
 sequence::UInt32,
 reward::Int32,
)::Nothing
 lock(trainer.lock) do
  if !isnothing(session.last_sequence)
   sequence == session.last_sequence + one(UInt32) ||
    throw(ArgumentError("out-of-order AI terminal sequence $sequence"))
  end
  previous_state = session.previous_state
  if is_training(trainer) && !isnothing(previous_state)
   previous_action = session.previous_action::Int
   _enqueue_transition_locked!(trainer, previous_state, previous_action, reward, nothing)
   log_event(
    "training_sample";
    session_id=previous_state[2],
    sequence,
    state=previous_state,
    legal_mask=legal_mask(previous_state),
    action=previous_action,
    action_name=action_name(previous_action),
    reward,
    next_state=nothing,
    next_legal_mask=nothing,
    terminal=true,
   )
  end
  session.previous_state = nothing
  session.previous_action = nothing
  session.last_sequence = nothing
  if is_training(trainer)
   _drain_pending_locked!(trainer)
   save_policy_checkpoint!(
    trainer.checkpoint_path,
    trainer.policy,
    trainer.optimizer_state,
    trainer.update_count,
   )
  end
 end
 return nothing
end

"""Compile the full gradient/update path on a disposable model before accepting training clients."""
function warmup_training_runtime!()::Nothing
 state = UInt32[
  2, 0, 0, 1_000, 500, 500, 20, 4, 8, 1, 1, 1, 1, 1, 4, 3,
  2, 1, 8, 5, 4, 2, 1, 1, 1, 1, 1, 1, 4, 3, 2, 1,
  UInt32(0x00000003), 0,
 ]
 trainer = create_trainer(
  mode=MODE_TRAIN,
  checkpoint_path=tempname(),
  batch_size=DEFAULT_BATCH_SIZE,
  checkpoint_every=typemax(Int),
 )
 action = select_action(trainer.policy, state)
 for _ in 1:DEFAULT_BATCH_SIZE
  enqueue_transition!(trainer, state, action, Int32(1), state)
 end
 return nothing
end


const DEFAULT_POLICY = create_policy()

select_action(state::AbstractVector{<:Integer}) = select_action(DEFAULT_POLICY, state)
