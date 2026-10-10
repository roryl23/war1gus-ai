# Shared by the standalone stdlib-only coordinator and the policy runtime.
function training_temperature_from_environment()::Union{Nothing,Float32}
 raw = get(ENV, "WAR1GUS_AI_TRAIN_TEMPERATURE", nothing)
 isnothing(raw) && return nothing
 value = tryparse(Float32, strip(raw))
 !isnothing(value) && isfinite(value) && value > 0.0f0 ||
  throw(ArgumentError("WAR1GUS_AI_TRAIN_TEMPERATURE must be a finite positive Float32"))
 return value
end
