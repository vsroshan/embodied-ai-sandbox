
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


def observe_environment(model, data):
    """Read the current simulated positions of objects."""

    observations = {}

    for name in OBJECT_NAMES:
        body_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            name
        )

        position = data.xpos[body_id].copy()

        observations[name] = position

    target_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "target_box"
    )

    target_position = model.geom_pos[target_id].copy()

    return observations, target_position


def print_observations(observations, target_position):
    """Display the observed environment state."""

    print("\n--- ENVIRONMENT OBSERVATION ---")

    for name, position in observations.items():
        print(
            f"{name}: "
            f"x={position[0]:.3f}, "
            f"y={position[1]:.3f}, "
            f"z={position[2]:.3f}"
        )

    print(
        "Target:",
        f"x={target_position[0]:.3f},",
        f"y={target_position[1]:.3f},",
        f"z={target_position[2]:.3f}"
    )

    print("-------------------------------")


def main():
    model = mujoco.MjModel.from_xml_string(MODEL_XML)
    data = mujoco.MjData(model)

    # Calculate the initial world-space positions.
    mujoco.mj_forward(model, data)

    print("Embodied AI Sandbox initialized!")
    print(f"MuJoCo version: {mujoco.__version__}")

    observations, target_position = observe_environment(
        model, data
    )

    print_observations(observations, target_position)

    print("\nLaunching 3D simulation...")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            mujoco.mj_step(model, data)

            # Synchronise the viewer with the simulation.
            viewer.sync()

            elapsed = time.time() - step_start
            remaining = model.opt.timestep - elapsed

            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
