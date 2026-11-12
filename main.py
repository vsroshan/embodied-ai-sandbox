
import time
import math

import mujoco
import mujoco.viewer


MODEL_XML = """
<mujoco model="embodied_ai_sandbox">
    <option gravity="0 0 -9.81" timestep="0.002"/>

    <visual>
        <headlight diffuse="0.8 0.8 0.8"
                   ambient="0.3 0.3 0.3"
                   specular="0.1 0.1 0.1"/>
    </visual>

    <worldbody>
        <geom name="ground"
              type="plane"
              pos="0 0 0"
              size="5 5 0.1"
              rgba="0.75 0.75 0.75 1"/>

        <body name="red_cube" pos="-0.6 0 0.08">
            <freejoint/>
            <geom name="red_cube_geom"
                  type="box"
                  size="0.08 0.08 0.08"
                  mass="0.2"
                  rgba="0.9 0.1 0.1 1"/>
        </body>

        <body name="blue_cube" pos="0 0 0.08">
            <freejoint/>
            <geom name="blue_cube_geom"
                  type="box"
                  size="0.08 0.08 0.08"
                  mass="0.2"
                  rgba="0.1 0.3 0.9 1"/>
        </body>

        <body name="green_cube" pos="0.6 0 0.08">
            <freejoint/>
            <geom name="green_cube_geom"
                  type="box"
                  size="0.08 0.08 0.08"
                  mass="0.2"
                  rgba="0.1 0.8 0.2 1"/>
        </body>

        <!-- Visual target marker: does not block cube movement. -->
        <geom name="target_box"
              type="box"
              pos="0 0.65 0.06"
              size="0.22 0.22 0.06"
              contype="0"
              conaffinity="0"
              rgba="0.9 0.75 0.1 0.5"/>
    </worldbody>
</mujoco>
"""


OBJECT_NAMES = ["red_cube", "blue_cube", "green_cube"]

# Target and cube footprint overlap threshold.
TARGET_REACH_THRESHOLD = 0.30

# PD controller parameters.
KP = 8.0
KD = 2.5
MAX_FORCE = 3.0

# Task evaluation parameters.
SETTLE_SPEED_THRESHOLD = 0.08
SETTLE_DURATION = 0.5
OBJECT_TIMEOUT = 25.0
TASK_TIMEOUT = 90.0


def observe_environment(model, data):
    """Read the current positions of all cubes and target."""

    observations = {}

    for name in OBJECT_NAMES:
        body_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            name
        )

        observations[name] = data.xpos[body_id].copy()

    target_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "target_box"
    )

    target_position = model.geom_pos[target_id].copy()

    return observations, target_position


def calculate_distance(position, target_position):
    """Calculate horizontal distance to target."""

    dx = position[0] - target_position[0]
    dy = position[1] - target_position[1]

    return math.sqrt(dx ** 2 + dy ** 2)


def evaluate_goal(observations, target_position):
    """Calculate distances between cubes and target."""

    distances = {}

    for name, position in observations.items():
        distances[name] = calculate_distance(
            position,
            target_position
        )

    return distances


def make_decision(distances, completed, failed):
    """Select the closest object that is not completed or failed."""

    eligible_objects = {
        name: distance
        for name, distance in distances.items()
        if name not in completed and name not in failed
    }

    if not eligible_objects:
        return None

    return min(eligible_objects, key=eligible_objects.get)


def is_inside_target(position, target_position):
    """Check whether the cube footprint overlaps the target."""

    dx = abs(position[0] - target_position[0])
    dy = abs(position[1] - target_position[1])

    return (
        dx <= TARGET_REACH_THRESHOLD
        and dy <= TARGET_REACH_THRESHOLD
    )


def get_linear_speed(data, body_id):
    """Read the magnitude of the body's linear velocity."""

    velocity = data.cvel[body_id, 3:6]

    return math.sqrt(
        velocity[0] ** 2
        + velocity[1] ** 2
        + velocity[2] ** 2
    )


def apply_controller(model, data, selected_object, target_position):
    """Apply PD control to move the selected cube."""

    if selected_object is None:
        return

    body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        selected_object
    )

    position = data.xpos[body_id]
    velocity = data.cvel[body_id, 3:6]

    error_x = target_position[0] - position[0]
    error_y = target_position[1] - position[1]

    force_x = KP * error_x - KD * velocity[0]
    force_y = KP * error_y - KD * velocity[1]

    force_magnitude = math.sqrt(
        force_x ** 2 + force_y ** 2
    )

    if force_magnitude > MAX_FORCE:
        scale = MAX_FORCE / force_magnitude
        force_x *= scale
        force_y *= scale

    data.xfrc_applied[body_id, 3:6] = [
        force_x,
        force_y,
        0.0
    ]


def print_progress(data, distances, selected_object,
                   completed, failed, elapsed_time):

    """Display live task execution information."""

    print("\n--- AUTONOMOUS TASK STATUS ---")

    for name, distance in distances.items():

        if name in completed:
            status = "COMPLETED"

        elif name in failed:
            status = "FAILED"

        elif name == selected_object:
            status = "CURRENTLY SELECTED"

        else:
            status = "WAITING"

        print(
            f"{name}: {distance:.3f} metres | {status}"
        )

    print(f"\nSelected object: {selected_object}")
    print(f"Completed: {len(completed)}/{len(OBJECT_NAMES)}")
    print(f"Failed: {len(failed)}")
    print(f"Simulation time: {data.time:.2f} seconds")
    print("--------------------------------")


def print_final_report(data, completed, failed,
                       completion_times, wall_time):

    """Generate the final task performance report."""

    total_objects = len(OBJECT_NAMES)
    completed_count = len(completed)
    failed_count = len(failed)

    completion_rate = (
        completed_count / total_objects
    ) * 100

    average_completion_time = (
        sum(completion_times.values()) / completed_count
        if completed_count > 0
        else 0
    )

    print("\n\n========== FINAL PERFORMANCE REPORT ==========")

    print(f"Total objects: {total_objects}")
    print(f"Successfully completed: {completed_count}")
    print(f"Failed objects: {failed_count}")

    print(f"Task completion rate: {completion_rate:.1f}%")

    print(f"Simulation duration: {data.time:.2f} seconds")
    print(f"Wall-clock duration: {wall_time:.2f} seconds")

    if completed_count > 0:
        print(
            "Average object completion time:",
            f"{average_completion_time:.2f} seconds"
        )

    print("\nCompleted objects:")

    for name, duration in completion_times.items():
        print(f"- {name}: {duration:.2f} seconds")

    print("\nFailed objects:")

    for name, reason in failed.items():
        print(f"- {name}: {reason}")

    if completed_count == total_objects:
        print("\nTASK RESULT: SUCCESS")
    elif completed_count > 0:
        print("\nTASK RESULT: PARTIAL SUCCESS")
    else:
        print("\nTASK RESULT: FAILURE")

    print("===============================================")


def main():

    model = mujoco.MjModel.from_xml_string(MODEL_XML)
    data = mujoco.MjData(model)

    mujoco.mj_forward(model, data)

    completed = set()
    failed = {}

    settle_timers = {
        name: 0.0 for name in OBJECT_NAMES
    }

    completion_times = {}

    selected_object = None
    object_start_time = 0.0

    last_decision_time = 0.0
    last_print_time = 0.0

    task_start_wall_time = time.time()

    task_finished = False
    task_result = "RUNNING"

    print("Embodied AI Sandbox initialized!")
    print(f"MuJoCo version: {mujoco.__version__}")
    print("Autonomous task execution started.")

    with mujoco.viewer.launch_passive(model, data) as viewer:

        while viewer.is_running():

            step_start = time.time()

            # Stop if the entire task exceeds its time limit.
            if data.time >= TASK_TIMEOUT:
                task_finished = True
                task_result = "TIMEOUT"
                break

            # Read the environment.
            observations, target_position = observe_environment(
                model, data
            )

            # Evaluate distances.
            distances = evaluate_goal(
                observations,
                target_position
            )

            # Check whether any object has settled inside the target.
            for name in OBJECT_NAMES:

                body_id = mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    name
                )

                position = observations[name]
                speed = get_linear_speed(data, body_id)

                if (
                    is_inside_target(position, target_position)
                    and speed <= SETTLE_SPEED_THRESHOLD
                ):

                    settle_timers[name] += model.opt.timestep

                else:
                    settle_timers[name] = 0.0

                if (
                    settle_timers[name] >= SETTLE_DURATION
                    and name not in completed
                    and name not in failed
                ):

                    completed.add(name)

                    completion_times[name] = (
                        data.time - object_start_time
                        if name == selected_object
                        else data.time
                    )

                    print(f"\nSUCCESS: {name} reached the target!")

                    if name == selected_object:
                        selected_object = None

            # Select a new object when necessary.
            if data.time - last_decision_time >= 0.2:

                if selected_object is None:

                    selected_object = make_decision(
                        distances,
                        completed,
                        failed
                    )

                    if selected_object is not None:
                        object_start_time = data.time
                        print(
                            f"\nSelected for movement: {selected_object}"
                        )

                last_decision_time = data.time

            # Check per-object timeout.
            if (
                selected_object is not None
                and data.time - object_start_time >= OBJECT_TIMEOUT
            ):

                failed[selected_object] = "Object movement timed out."

                print(
                    f"\nFAILED: {selected_object} exceeded its time limit."
                )

                selected_object = None

            # Clear previous forces and apply current action.
            data.xfrc_applied[:] = 0

            apply_controller(
                model,
                data,
                selected_object,
                target_position
            )

            # Advance physics.
            mujoco.mj_step(model, data)

            # Print progress once per simulation second.
            if data.time - last_print_time >= 1.0:

                print_progress(
                    data,
                    distances,
                    selected_object,
                    completed,
                    failed,
                    data.time
                )

                last_print_time = data.time

            # Stop when every object is completed or failed.
            if len(completed) + len(failed) == len(OBJECT_NAMES):

                task_finished = True
                task_result = "FINISHED"
                break

            viewer.sync()

            elapsed = time.time() - step_start
            remaining = model.opt.timestep - elapsed

            if remaining > 0:
                time.sleep(remaining)

    wall_time = time.time() - task_start_wall_time

    if task_finished:
        print(f"\nTask execution ended: {task_result}")
        print_final_report(
            data,
            completed,
            failed,
            completion_times,
            wall_time
        )
    else:
        print("\nSimulation viewer closed before task completion.")


if __name__ == "__main__":
    main()
