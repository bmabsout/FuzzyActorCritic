from dataclasses import dataclass
import pybullet as p
import time
import numpy as np
import utils
import argparse


@dataclass
class Point:
    x: float
    y: float

@dataclass
class World:
    """A dataclass to hold the world objects"""
    plate: int
    sphere: int

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setpoint", type=float, nargs=2, default=(0.0,0.0))
    parser.add_argument("--kp", type=float, default=1.0)
    parser.add_argument("--kd", type=float, default=50.0)
    parser.add_argument("--noise", action="store_true", help="Add noise to the measurements")
    parser.add_argument("--filtered", action="store_true", help="filter the measurements")
    cmd_args = parser.parse_args()
    print(cmd_args)
    cmd_args.setpoint = Point(*cmd_args.setpoint)
    return cmd_args

class PD:
    def __init__(self, kp, kd):
        self.prev_error = 0.0
        self.kp = kp
        self.kd = kd
    def pd(self, error):
        error_diff =  error - self.prev_error
        self.prev_error = error
        return np.clip(error*self.kp + error_diff*self.kd, -0.1, 0.1)

def run_controller(kp, kd, setpoint, noise, filtered, world: World):
    pd_x = PD(kp=kp, kd=kd)
    pd_y = PD(kp=kp, kd=kd)
    x_filter = utils.Butterworth_filter(butterworth_order=5, cutoff_freq=20)
    y_filter = utils.Butterworth_filter(butterworth_order=5, cutoff_freq=20)
    def set_plate_angles(theta_x, theta_y):
        p.setJointMotorControl2(world.plate, 1, p.POSITION_CONTROL, targetPosition=np.clip(theta_x, -0.1, 0.1), force=5, maxVelocity=2)
        p.setJointMotorControl2(world.plate, 0, p.POSITION_CONTROL, targetPosition=np.clip(-theta_y, -0.1, 0.1), force=5, maxVelocity=2)

    def produce_forces(x,y):
        return (pd_x.pd(setpoint.x - x), pd_y.pd(setpoint.y - y))

    def every_10ms(i: int, t: float):
        '''This function is called every ms and performs the following:
        1. Get the measurement of the position of the ball
        2. Calculate the forces to be applied to the plate
        3. Apply the forces to the plate
        '''
        (x,y,z), orientation = p.getBasePositionAndOrientation(world.sphere)
        if noise:
            x += utils.noise(t)
            y += utils.noise(t, seed = 43) # so that the noise on y is different than the one on x
        
        if filtered:
            x = x_filter.step(x)
            y = y_filter.step(y)
        (angle_x, angle_y) = produce_forces(x, y)
        set_plate_angles(angle_x, angle_y)

        if i%10 == 0:
            print(f"t: {t:.2f}, x: {x:.3f},\ty: {y:.3f},\tax: {angle_x:.3f},\tay: {angle_y:.3f}")

    utils.loop_every(0.01, every_10ms)

def run_simulation( initial_ball_position = Point(np.random.uniform(-0.2, 0.2),
                                                  np.random.uniform(-0.2, 0.2))):
    p.connect(p.GUI)
    p.setAdditionalSearchPath("assets")
    plate = p.loadURDF("plate.urdf")

    #zoom to the plate
    p.resetDebugVisualizerCamera(cameraDistance=1.0, cameraYaw=0, cameraPitch=-45, cameraTargetPosition=[0,0,0])

    p.setJointMotorControl2(plate, 0, p.POSITION_CONTROL, targetPosition=0, force=5, maxVelocity=2)
    p.setJointMotorControl2(plate, 1, p.POSITION_CONTROL, targetPosition=0, force=5, maxVelocity=2)

    p.setGravity(0, 0, -9.8)
    sphere = p.createMultiBody(0.2
        , p.createCollisionShape(p.GEOM_SPHERE, radius=0.04)
        , basePosition = [initial_ball_position.x,initial_ball_position.y,0.5]
    )
    #update the simulation at 100 Hz
    p.setTimeStep(0.01)
    p.setRealTimeSimulation(1)
    return World(plate=plate, sphere=sphere)


if __name__ == "__main__":
    cmd_args = parse_args()
    world = run_simulation()
    run_controller(**vars(cmd_args), world=world)
    time.sleep(10000)
