import mujoco 
import mujoco.viewer
import time 

model = mujoco.MjModel.from_xml_path("engine/robotics/simulation/mujoco/models/humanoid/toy_humanoid.xml") ## Contains Robots config or structure

data = mujoco.MjData(model) ## Contains current state of the robot - joint pos, vel, actuator controls...

joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "shoulder") ## FInd the ID of that joint shoulder
qpos_addr = model.jnt_qposadr[joint_id]  ## Find the respective qpos address for that joint ID

actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "shoulder_motor")

target_angle = 0.8 ## 0.8 rad for that joint to rotate, passing as control to the motor or actuator

print("joint_id:", joint_id)
print("qpos_addr:", qpos_addr)
print("actuator_id:", actuator_id)

print("initial qpos:", data.qpos[qpos_addr])


print("joint type:", model.jnt_type[joint_id])
print("joint axis:", model.jnt_axis[joint_id])
with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():
        data.ctrl[actuator_id] = target_angle ## Control isnide the loop makes sure the control input actually update
        print(f"targe: {target_angle}, qpos: {data.qpos[qpos_addr]}, qvel: {data.qvel[qpos_addr]}, actuator_force: {data.qfrc_actuator[qpos_addr]} \n")
        print("ctrl:", data.ctrl[actuator_id])
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)