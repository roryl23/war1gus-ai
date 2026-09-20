using Flux
using Random
using Statistics

const STATE_DIM = 18
const ACTION_DIM = 10
const EMBED_DIM = 48
const NUM_HEADS = 4
const HIDDEN_DIM = 96
const TOKEN_COUNT = 3

const ACTION_ECONOMY = 0
const ACTION_SUPPLY = 1
const ACTION_INFRASTRUCTURE = 2
const ACTION_BLACKSMITH = 3
const ACTION_BASIC_FORCE = 4
const ACTION_CAVALRY_FORCE = 5
const ACTION_SIEGE_FORCE = 6
const ACTION_ATTACK = 7
const ACTION_RESEARCH = 8
const ACTION_DEFEND = 9

"""Small state encoder followed by one multi-head self-attention transformer block."""
struct AiPolicy
  state_encoder
  token_encoder
  attention
  feed_forward
  action_head
end

function create_policy(; seed::Integer=0x574131)
  rng = MersenneTwister(seed)
  init = (dimensions...) -> Flux.glorot_uniform(rng, dimensions...)
  return AiPolicy(
    Dense(STATE_DIM => EMBED_DIM, tanh; init),
    Dense(EMBED_DIM => TOKEN_COUNT * EMBED_DIM, tanh; init),
    Flux.MultiHeadAttention(EMBED_DIM; nheads=NUM_HEADS, dropout_prob=0.0f0, init),
    Chain(Dense(EMBED_DIM => HIDDEN_DIM, relu; init), Dense(HIDDEN_DIM => EMBED_DIM; init)),
    Dense(EMBED_DIM => ACTION_DIM; init),
  )
end

const DEFAULT_POLICY = create_policy()

function validate_state(state::AbstractVector{<:Integer})::Nothing
  length(state) == STATE_DIM || throw(ArgumentError("AI state has $(length(state)) fields, expected $STATE_DIM"))
  UInt32(state[1]) == UInt32(1) || throw(ArgumentError("unsupported AI state protocol version $(state[1])"))
  return nothing
end

"""Map unsigned protocol counters to a bounded, compact model input."""
function encode_state(state::AbstractVector{<:Integer})::Vector{Float32}
  validate_state(state)
  values = Float32.(state)
  scales = Float32[
    1, 7, 1, 20_000, 2_000, 2_000, 100, 100, 50,
    10, 10, 10, 10, 10, 50, 50, 50, 50,
  ]
  return clamp.(values ./ scales, 0.0f0, 4.0f0)
end

function action_logits(policy::AiPolicy, state::AbstractVector{<:Integer})::Vector{Float32}
  encoded = policy.state_encoder(encode_state(state))
  tokens = reshape(policy.token_encoder(encoded), EMBED_DIM, TOKEN_COUNT, 1)
  attended = policy.attention(tokens)[1]
  transformed = tokens .+ attended
  transformed .+= policy.feed_forward(transformed)
  pooled = vec(mean(transformed; dims=(2, 3)))
  return clamp.(Float32.(policy.action_head(pooled)), -1.0f0, 1.0f0)
end

@inline state_count(state::AbstractVector{<:Integer}, index::Integer) = Int(state[index])

"""Mask unavailable actions and return small biases among strategic choices."""
function action_priors(state::AbstractVector{<:Integer})::Vector{Float32}
  validate_state(state)
  scores = fill(-Inf32, ACTION_DIM)

  gold = state_count(state, 5)
  wood = state_count(state, 6)
  supply = state_count(state, 7)
  demand = state_count(state, 8)
  workers = state_count(state, 9)
  barracks = state_count(state, 11)
  lumber_mills = state_count(state, 12)
  blacksmiths = state_count(state, 13)
  stables = state_count(state, 14)
  soldiers = state_count(state, 15)
  shooters = state_count(state, 16)
  cavalry = state_count(state, 17)
  catapults = state_count(state, 18)
  basic_force = soldiers + shooters
  army = basic_force + cavalry + catapults

  if demand + 2 >= supply
    scores[ACTION_SUPPLY+1] = 0.0f0
    return scores
  elseif workers < 5 || (gold < 250 && wood < 100)
    scores[ACTION_ECONOMY+1] = 0.0f0
    return scores
  elseif barracks == 0 || lumber_mills == 0
    scores[ACTION_INFRASTRUCTURE+1] = 0.0f0
    return scores
  elseif basic_force < 4
    scores[ACTION_BASIC_FORCE+1] = 0.0f0
    return scores
  elseif blacksmiths == 0
    scores[ACTION_BLACKSMITH+1] = 0.0f0
    return scores
  elseif stables == 0
    scores[ACTION_INFRASTRUCTURE+1] = 0.0f0
    return scores
  end

  scores[ACTION_ECONOMY+1] = 0.0f0
  scores[ACTION_SUPPLY+1] = -0.2f0
  scores[ACTION_BASIC_FORCE+1] = 0.1f0
  scores[ACTION_CAVALRY_FORCE+1] = cavalry < 2 ? 0.2f0 : 0.0f0
  scores[ACTION_SIEGE_FORCE+1] = catapults < 1 ? 0.2f0 : 0.0f0
  scores[ACTION_RESEARCH+1] = 0.0f0
  scores[ACTION_DEFEND+1] = 0.0f0
  army >= 6 && (scores[ACTION_ATTACK+1] = 0.2f0)
  return scores
end

"""Combine state-conditioned transformer logits with legal progression priors."""
function select_action(policy::AiPolicy, state::AbstractVector{<:Integer})::Int
  scores = action_logits(policy, state) .+ action_priors(state)
  return argmax(scores) - 1
end

select_action(state::AbstractVector{<:Integer}) = select_action(DEFAULT_POLICY, state)
