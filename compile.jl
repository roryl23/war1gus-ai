using PackageCompiler

unknown_args = filter(!=("--incremental"), ARGS)
isempty(unknown_args) || error("unknown arguments: $(join(unknown_args, ' '))")
incremental = "--incremental" in ARGS

@time "War1gusAI compilation" PackageCompiler.create_app(
  ".",
  "build",
  force=true,
  cpu_target="native",
  precompile_execution_file="precompile.jl",
  incremental=incremental,
)
