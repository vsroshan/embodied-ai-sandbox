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
        <!-- Ground -->
        <geom name="ground"
              type="plane"
              pos="0 0 0"
              size="5 5 0.1"
              rgba="0.75 0.75 0.75 1"/>

        <!-- Red cube -->
        <body name="red_cube" pos="-0.6 0 0.08">
            <freejoint/>
            <geom name="red_cube_geom"
                  type="box"
                  size="0.08 0.08 0.08"
                  mass="0.2"
                  rgba="0.9 0.1 0.1 1"/>
        </body>

        <!-- Blue cube -->
        <body name="blue_cube" pos="0 0 0.08">
            <freejoint/>
            <geom name="blue_cube_geom"
                  type="box"
                  size="0.08 0.08 0.08"
                  mass="0.2"
                  rgba="0.1 0.3 0.9 1"/>
        </body>

        <!-- Green cube -->
        <body name="green_cube" pos="0.6 0 0.08">
            <freejoint/>
            <geom name="green_cube_geom"
                  type="box"
                  size="0.08 0.08 0.08"
                  mass="0.2"
                  rgba="0.1 0.8 0.2 1"/>
        </body>

        <!-- Target zone -->
        <geom name="target_box"
              type="box"
              pos="0 0.65 0.06"
              size="0.22 0.22 0.06"
              rgba="0.9 0.75 0.1 1"/>
    </worldbody>
</mujoco>
"""


def main():
    model = mujoco.MjModel.from_xml_string(MODEL_XML)
    data = mujoco.MjData(model)

    print("Embodied AI Sandbox initialized!")
    print(f"MuJoCo version: {mujoco.__version__}")
    print(f"Number of bodies: {model.nbody}")
    print("Close the simulation window to exit.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            mujoco.mj_step(model, data)
            viewer.sync()

            # Keep simulation speed close to real time.
            elapsed = time.time() - step_start
            remaining = model.opt.timestep - elapsed

            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()