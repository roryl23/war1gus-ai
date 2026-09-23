using War1gusAI
using Sockets

# Exercise the application on disposable v3 data, including the real TCP accept,
# frame decoding, per-player session, response, and terminal paths.
function warmup_inference_runtime!()
    mktempdir() do directory
        trainer = War1gusAI.create_trainer(
            checkpoint_path=joinpath(directory, "policy.jls"),
            league_path=joinpath(directory, "league"),
            league_snapshot_override=nothing,
            read_only=true,
            seed=7,
        )
        state = UInt32[
            3, 0, 0, 300, 500, 500, 20, 4, 128, 128, 32, 32,
            1_000, 1_000, 500, 500, 0, 0, 0, 0, 0, 0,
        ]
        for index in 1:32
            append!(state, UInt32[index, 0x1234 + index, 1, 1, 20, 20, 60, 60, 400, 0, 0, 0, 1, 4])
        end
        append!(state, zeros(UInt32, 12)) # The mandatory first candidate is wait.
        for index in 1:31
            append!(state, UInt32[index % 12 + 1, index, (index % 32) + 1, 0, 20, 20, 1, 0, 5, 10, 1, index])
        end

        function frame(prefix::UInt8, sequence::UInt32, state::Vector{UInt32})
            io = IOBuffer()
            write(io, prefix)
            War1gusAI.write_u32_be(io, sequence)
            War1gusAI.write_u32_be(io, UInt32(0))
            War1gusAI.write_u32_be(io, UInt32(length(state)))
            War1gusAI.write_u32_be(io, state[12])
            for word in state
                War1gusAI.write_u32_be(io, word)
            end
            return take!(io)
        end

        step = frame(UInt8('S'), UInt32(0), state)
        decoded = War1gusAI.decode_step_frame(IOBuffer(step[2:end]), UInt8('S'))
        isnothing(decoded) && error("warmup step did not decode")
        expected = War1gusAI.process_step!(
            trainer, War1gusAI.ClientSession(), UInt32(0), Int32(0), decoded.state,
        )
        terminal = copy(state)
        resize!(terminal, 22 + 32 * 14)
        terminal[12] = 0

        listener = listen(ip"127.0.0.1", 0)
        port = Int(getsockname(listener)[2])
        server_task = @async War1gusAI.serve(
            "127.0.0.1", port; trainer, monitor_stdin=false, listener,
        )
        client = nothing
        try
            client = connect(ip"127.0.0.1", port)
            write(client, UInt8['I', 3])
            for sequence in 0:1
                write(client, frame(UInt8('S'), UInt32(sequence), state))
                flush(client)
                response = War1gusAI.read_exact(client, 4)
                isnothing(response) && error("warmup handler closed before responding")
                action = War1gusAI.decode_u32_be(response, 1)
                action == expected || error("warmup handler returned a different action")
            end
            write(client, frame(UInt8('E'), UInt32(2), terminal))
            flush(client)
            eof(client) || error("warmup handler did not close after terminal frame")
        finally
            !isnothing(client) && isopen(client) && close(client)
            isopen(listener) && close(listener)
            wait(server_task) # Includes the handler task and its session cleanup.
        end
    end
    return nothing
end

warmup_inference_runtime!()

War1gusAI.warmup_training_runtime!()
