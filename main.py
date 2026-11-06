
import time

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

# Target half-width is 0.22 m.
# Cube half-width is 0.08 m.
# Their footprints overlap when their center separation
# along both X and Y is at most 0.22 + 0.08 = 0.30 m.
TARGET_REACH_THRESHOLD = 0.30


def observe_environment(model, data):
    """Read the current simulated positions of objects."""

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
    """Calculate horizontal XY distance from an object to target."""

    dx = position[0] - target_position[0]
    dy = position[1] - target_position[1]

    return (dx ** 2 + dy ** 2) ** 0.5


def evaluate_goal(observations, target_position):
    """Evaluate distances, closest object, and target reach."""

    distances = {}

    for name, position in observations.items():
        distances[name] = calculate_distance(
            position,
            target_position
        )

    closest_object = min(distances, key=distances.get)

    objects_at_target = []

    for name, position in observations.items():
        dx = abs(position[0] - target_position[0])
        dy = abs(position[1] - target_position[1])

        # Check whether the cube footprint overlaps the
        # rectangular target footprint.
        if (
            dx <= TARGET_REACH_THRESHOLD
            and dy <= TARGET_REACH_THRESHOLD
        ):
            objects_at_target.append(name)

    return distances, closest_object, objects_at_target


def print_evaluation(distances, closest_object, objects_at_target):
    """Display the goal evaluation results."""

    print("\n--- GOAL EVALUATION ---")

    for name, distance in distances.items():
        print(f"{name}: {distance:.3f} metres from target")

    print(f"\nClosest object: {closest_object}")

    if objects_at_target:
        print("Target reached by:", ", ".join(objects_at_target))
    else:
        print("Target status: No object has reached the target.")

    print("-----------------------")


def main():
    model = mujoco.MjModel.from_xml_string(MODEL_XML)
    data = mujoco.MjData(model)

    mujoco.mj_forward(model, data)

    print("Embodied AI Sandbox initialized!")
    print(f"MuJoCo version: {mujoco.__version__}")

    print("\nLaunching 3D simulation...")

    last_evaluation_time = 0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            mujoco.mj_step(model, data)

            current_time = time.time()

            # Evaluate approximately once per second.
            if current_time - last_evaluation_time >= 1.0:
                observations, target_position = observe_environment(
                    model, data
                )

                distances, closest_object, objects_at_target = (
                    evaluate_goal(observations, target_position)
                )

                print_evaluation(
                    distances,
                    closest_object,
                    objects_at_target
                )

                last_evaluation_time = current_time

            viewer.sync()

            elapsed = time.time() - step_start
            remaining = model.opt.timestep - elapsed

            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
