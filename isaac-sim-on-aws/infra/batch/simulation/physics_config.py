"""Physics parameter configuration for Isaac Sim."""


def configure_physics(world, config: dict):
    """Apply physics settings from config to the simulation world."""
    physics_context = world.get_physics_context()

    # Solver
    solver = config.get("solver", "TGS")
    physics_context.set_solver_type(solver)
    physics_context.set_position_iteration_count(
        config.get("position_iterations", 16)
    )
    physics_context.set_velocity_iteration_count(
        config.get("velocity_iterations", 1)
    )

    # Gravity
    gravity = config.get("gravity", [0.0, 0.0, -9.81])
    physics_context.set_gravity(gravity)

    print(
        f"Physics: {solver} solver, "
        f"{config.get('position_iterations', 16)} iterations, "
        f"gravity={gravity}"
    )
