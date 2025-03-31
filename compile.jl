using ArgParse
using PackageCompiler

s = ArgParseSettings()
@add_arg_table! s begin
  "--incremental"
    help = "enable incremental compilation"
    action = :store_true
end
parsed_args = parse_args(ARGS, s)

@time "War1gusAI compilation" PackageCompiler.create_app(
  ".",
  "build/bin/war1gus-ai",
  force=true,
  incremental=parsed_args["incremental"],
)
