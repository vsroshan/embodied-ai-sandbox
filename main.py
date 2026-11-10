
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

        <geom name="target_box"
              type="box"
              pos="0 0.65 0.06"
              size="0.22 0.22 0.06"
              rgba="0.9 0.75 0.1 1"/>
    </worldbody>
</mujoco>
"""


OBJECT_NAMES = ["red_cube", "blue_cube", "green_cube"]

TARGET_REACH_THRESHOLD = 0.30

# Controller parameters
KP = 8.0
KD = 2.5
MAX_FORCE = 3.0


def observe_environment(model, data):
    """Read the current simulated positions."""

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
    """Calculate horizontal distance to the target."""

    dx = position[0] - target_position[0]
    dy = position[1] - target_position[1]

    return math.sqrt(dx ** 2 + dy ** 2)


def evaluate_goal(observations, target_position):
    """Evaluate distances and target reach."""

    distances = {}
    objects_at_target = []

    for name, position in observations.items():

        distances[name] = calculate_distance(
            position,
            target_position
        )

        dx = abs(position[0] - target_position[0])
        dy = abs(position[1] - target_position[1])

        if (
            dx <= TARGET_REACH_THRESHOLD
            and dy <= TARGET_REACH_THRESHOLD
        ):
            objects_at_target.append(name)

    closest_object = min(distances, key=distances.get)

    return distances, closest_object, objects_at_target


def make_decision(distances, objects_at_target):
    """Select the closest object that has not reached the target."""

    eligible_objects = {
        name: distance
        for name, distance in distances.items()
        if name not in objects_at_target
    }

    if not eligible_objects:
        return {
            "action": "NO_ACTION_REQUIRED",
            "selected_object": None,
            "reason": "All objects have reached the target."
        }

    selected_object = min(
        eligible_objects,
        key=eligible_objects.get
    )

    return {
        "action": "MOVE_OBJECT",
        "selected_object": selected_object,
        "reason": (
            f"{selected_object} is the closest object "
            "that has not reached the target."
        )
    }


def apply_controller(model, data, selected_object, target_position):
    """
    Apply a PD controller to move the selected cube
    horizontally towards the target.
    """

    if selected_object is None:
        return

    body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        selected_object
    )

    position = data.xpos[body_id]

    # MuJoCo cvel stores angular velocity first,
    # followed by linear velocity in world coordinates.
    velocity = data.cvel[body_id, 3:6]

    error_x = target_position[0] - position[0]
    error_y = target_position[1] - position[1]

    # PD control: proportional position correction
    # plus damping to reduce overshoot.
    force_x = KP * error_x - KD * velocity[0]
    force_y = KP * error_y - KD * velocity[1]

    # Limit the total horizontal force.
    force_magnitude = math.sqrt(
        force_x ** 2 + force_y ** 2
    )

    if force_magnitude > MAX_FORCE:
        scale = MAX_FORCE / force_magnitude
        force_x *= scale
        force_y *= scale

    # Apply force in the world X and Y directions.
    data.xfrc_applied[body_id, 3:6] = [
        force_x,
        force_y,
        0.0
    ]


def print_status(distances, closest_object, objects_at_target, decision):
    """Display evaluation and controller status."""

    print("\n--- ACTION & CONTROL STATUS ---")

    for name, distance in distances.items():
        status = (
            "AT TARGET"
            if name in objects_at_target
            else "NOT AT TARGET"
        )

        print(f"{name}: {distance:.3f} metres | {status}")

    print(f"\nClosest object: {closest_object}")
    print("Action:", decision["action"])
    print("Selected object:", decision["selected_object"])
    print("Reason:", decision["reason"])

    print("--------------------------------")


def main():
    model = mujoco.MjModel.from_xml_string(MODEL_XML)
    data = mujoco.MjData(model)

    mujoco.mj_forward(model, data)

    print("Embodied AI Sandbox initialized!")
    print(f"MuJoCo version: {mujoco.__version__}")
    print("Action and control system ready.")

    decision = {
        "action": "NO_ACTION_REQUIRED",
        "selected_object": None,
        "reason": "Waiting for initial evaluation."
    }

    last_decision_time = 0
    last_print_time = 0

    with mujoco.viewer.launch_passive(model, data) as viewer:

        while viewer.is_running():

            step_start = time.time()
            current_time = time.time()

            # Refresh the decision every 0.2 seconds.
            if current_time - last_decision_time >= 0.2:

                observations, target_position = observe_environment(
                    model, data
                )

                distances, closest_object, objects_at_target = (
                    evaluate_goal(observations, target_position)
                )

                decision = make_decision(
                    distances,
                    objects_at_target
                )

                last_decision_time = current_time

            # Clear previously applied external forces.
            data.xfrc_applied[:] = 0

            # Execute the current decision using the controller.
            apply_controller(
                model,
                data,
                decision["selected_object"],
                target_position
            )

            # Advance the physics simulation.
            mujoco.mj_step(model, data)

            # Print status approximately once per second.
            if current_time - last_print_time >= 1.0:

                observations, target_position = observe_environment(
                    model, data
                )

                distances, closest_object, objects_at_target = (
                    evaluate_goal(observations, target_position)
                )

                print_status(
                    distances,
                    closest_object,
                    objects_at_target,
                    decision
                )

                last_print_time = current_time

            viewer.sync()

            elapsed = time.time() - step_start
            remaining = model.opt.timestep - elapsed

            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()

