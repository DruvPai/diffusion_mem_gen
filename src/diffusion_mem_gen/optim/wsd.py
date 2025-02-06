import optax


def wsd(
    init_lr: float,
    peak_lr: float,
    final_lr: float,
    num_steps: int,
    num_steps_warmup: int,
    num_steps_decay: int,
) -> optax.Schedule:
    return optax.join_schedules(
        [
            optax.linear_schedule(
                init_value=init_lr, end_value=peak_lr, transition_steps=num_steps_warmup
            ),
            optax.linear_schedule(
                init_value=peak_lr,
                end_value=peak_lr,
                transition_steps=num_steps - num_steps_warmup - num_steps_decay,
            ),
            optax.linear_schedule(
                init_value=peak_lr, end_value=final_lr, transition_steps=num_steps_decay
            ),
        ],
        boundaries=[num_steps_warmup, num_steps - num_steps_decay],
    )
