import json
import os
import socket
import sys
import threading
import time
import traceback

DEBUG = os.environ.get("OAK_CARLA_DEBUG") == "1"
VEHICLE_ROLE_NAME = "oak_vehicle"
CAMERA_ROLE_NAME = "oak_camera"
CARLA_IMAGE_SIZE = max(int(os.environ.get("OAK_CARLA_IMAGE_SIZE", "4")), 4)


def encode_message(message):
    return json.dumps(message, separators=(",", ":")) + "\n"


def emit(message):
    sys.stdout.write(encode_message(message))
    sys.stdout.flush()


def send_message(connection, message):
    connection.sendall(encode_message(message).encode("utf-8"))


def debug(message):
    if not DEBUG:
        return
    sys.stderr.write(f"[carla_whisky_bridge] {message}\n")
    sys.stderr.flush()


def install_carla_paths(egg_path, python_api_path):
    for path in [egg_path, python_api_path]:
        if path and path not in sys.path:
            sys.path.insert(0, path)


def normalize_rgb(raw_bytes):
    rgb = []
    for index in range(0, len(raw_bytes), 4):
        blue = raw_bytes[index]
        green = raw_bytes[index + 1]
        red = raw_bytes[index + 2]
        rgb.extend((red / 255.0, green / 255.0, blue / 255.0))
    return rgb


class CarlaBridge:
    def __init__(self, carla_module, host, port):
        self.carla = carla_module
        self.client = carla_module.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.original_settings = self.world.get_settings()
        self.vehicle = None
        self.camera = None
        self.image = None
        self.image_lock = threading.Lock()
        self.obs_dim = CARLA_IMAGE_SIZE * CARLA_IMAGE_SIZE * 3
        self.act_dim = 2

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)
        debug(f"Connected to CARLA world at {host}:{port} and enabled synchronous mode")

    def reset(self):
        debug("Reset: destroying existing actors")
        destroyed_actors = self._destroy_actors()
        debug("Reset: destroying stale OaK actors")
        destroyed_actors = self._destroy_stale_actors() or destroyed_actors
        if destroyed_actors:
            debug("Reset: world tick after actor cleanup")
            self.world.tick()
        blueprint = self.world.get_blueprint_library().find("vehicle.tesla.model3")
        blueprint.set_attribute("role_name", VEHICLE_ROLE_NAME)
        self.vehicle = self._spawn_vehicle(blueprint)
        debug("Reset: spawned vehicle")

        camera_blueprint = self.world.get_blueprint_library().find("sensor.camera.rgb")
        camera_blueprint.set_attribute("role_name", CAMERA_ROLE_NAME)
        camera_blueprint.set_attribute("image_size_x", str(CARLA_IMAGE_SIZE))
        camera_blueprint.set_attribute("image_size_y", str(CARLA_IMAGE_SIZE))

        transform = self.carla.Transform(self.carla.Location(x=1.5, z=2.4))
        self.camera = self.world.spawn_actor(camera_blueprint, transform, attach_to=self.vehicle)
        debug("Reset: spawned camera")
        self._clear_image()
        self.camera.listen(self._on_image)
        debug("Reset: camera listening, ticking world")
        self.world.tick()
        debug("Reset: world tick returned, waiting for image")

        return {"status": "ok", "image": self._wait_for_image()}

    def step(self, action):
        if self.vehicle is None:
            raise RuntimeError("CARLA vehicle is not initialized; call reset first")

        steer = clamp(action[0], -1.0, 1.0)
        throttle_axis = clamp(action[1], -1.0, 1.0)
        debug(f"Step: steer={steer:.3f} throttle_axis={throttle_axis:.3f}")

        control = self.carla.VehicleControl()
        control.steer = steer
        control.throttle = max(0.0, throttle_axis)
        control.brake = max(0.0, -throttle_axis)

        self.vehicle.apply_control(control)
        self._clear_image()
        debug("Step: control applied, ticking world")
        self.world.tick()
        debug("Step: world tick returned, waiting for image")

        velocity = self.vehicle.get_velocity()
        speed = (velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) ** 0.5
        debug(f"Step: speed={speed:.3f}")

        return {
            "status": "ok",
            "image": self._wait_for_image(),
            "reward": speed * 0.01,
            "done": False,
        }

    def close(self):
        debug("Close: destroying actors")
        destroyed = self._destroy_actors()
        if destroyed:
            debug("Close: world tick after actor cleanup")
            self.world.tick()
        debug("Close: restoring original settings")
        self.world.apply_settings(self.original_settings)

    def _clear_image(self):
        with self.image_lock:
            self.image = None

    def _on_image(self, image):
        raw = bytes(image.raw_data)
        normalized = normalize_rgb(raw)
        with self.image_lock:
            self.image = normalized
        debug("Received camera frame")

    def _wait_for_image(self, timeout=5.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self.image_lock:
                if self.image is not None:
                    return self.image
            time.sleep(0.01)
        raise TimeoutError("Timed out waiting for a CARLA camera frame")

    def _destroy_actors(self):
        destroyed = False
        if self.camera is not None:
            try:
                self.camera.stop()
            except Exception:
                pass
            self.camera.destroy()
            self.camera = None
            destroyed = True
        if self.vehicle is not None:
            self.vehicle.destroy()
            self.vehicle = None
            destroyed = True
        return destroyed

    def _destroy_stale_actors(self):
        destroyed = False
        for actor in self.world.get_actors():
            role_name = actor.attributes.get("role_name")
            if role_name not in {VEHICLE_ROLE_NAME, CAMERA_ROLE_NAME}:
                continue
            try:
                actor.destroy()
                destroyed = True
            except RuntimeError:
                pass
        return destroyed

    def _spawn_vehicle(self, blueprint):
        for spawn_point in self.world.get_map().get_spawn_points():
            vehicle = self.world.try_spawn_actor(blueprint, spawn_point)
            if vehicle is not None:
                return vehicle
        raise RuntimeError("Failed to spawn OaK vehicle at any CARLA spawn point")


def clamp(value, lower, upper):
    return max(lower, min(upper, float(value)))


def connect_world(carla_module, host, port):
    client = carla_module.Client(host, port)
    client.set_timeout(10.0)
    world = client.get_world()
    return client, world


def world_settings_payload(world):
    settings = world.get_settings()
    fixed_delta_seconds = settings.fixed_delta_seconds
    if fixed_delta_seconds is not None:
        fixed_delta_seconds = float(fixed_delta_seconds)
    return {
        "status": "ok",
        "synchronous_mode": bool(settings.synchronous_mode),
        "fixed_delta_seconds": fixed_delta_seconds,
    }


def release_world_settings(world):
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)
    return world_settings_payload(world)


def handle_command(bridge, request):
    command = request.get("cmd")

    if command == "reset":
        return bridge.reset(), False

    if command == "step":
        return bridge.step(request.get("action", [0.0, 0.0])), False

    if command == "close":
        bridge.close()
        return {"status": "closed"}, True

    return {"status": "error", "error": f"unknown command: {command}"}, True


def main():
    if len(sys.argv) < 4:
        emit({"status": "error", "error": "usage: bridge.py <mode> <carla-egg> <carla-python-api> [host] [port]"})
        return 2

    mode = sys.argv[1]
    egg_path = sys.argv[2]
    python_api_path = sys.argv[3]
    listen_host = "127.0.0.1"
    listen_port = None
    host = "127.0.0.1"
    port = 2000

    install_carla_paths(egg_path, python_api_path)

    try:
        import carla
    except Exception as error:
        emit({"status": "error", "error": str(error), "traceback": traceback.format_exc()})
        return 1

    if mode == "healthcheck":
        emit({"status": "ok", "module": carla.__file__})
        return 0

    if mode == "settings":
        host = sys.argv[4] if len(sys.argv) > 4 else "127.0.0.1"
        port = int(sys.argv[5]) if len(sys.argv) > 5 else 2000
        _, world = connect_world(carla, host, port)
        emit(world_settings_payload(world))
        return 0

    if mode == "release":
        host = sys.argv[4] if len(sys.argv) > 4 else "127.0.0.1"
        port = int(sys.argv[5]) if len(sys.argv) > 5 else 2000
        _, world = connect_world(carla, host, port)
        emit(release_world_settings(world))
        return 0

    if mode == "serve":
        listen_host = sys.argv[4] if len(sys.argv) > 4 else "127.0.0.1"
        listen_port = int(sys.argv[5]) if len(sys.argv) > 5 else 23000
        host = sys.argv[6] if len(sys.argv) > 6 else "127.0.0.1"
        port = int(sys.argv[7]) if len(sys.argv) > 7 else 2000
    elif mode == "bridge":
        host = sys.argv[4] if len(sys.argv) > 4 else "127.0.0.1"
        port = int(sys.argv[5]) if len(sys.argv) > 5 else 2000
    else:
        emit({"status": "error", "error": f"unknown mode: {mode}"})
        return 2

    bridge = None
    active_connection = None
    try:
        bridge = CarlaBridge(carla, host, port)
        emit({"status": "ready", "obs_dim": bridge.obs_dim, "act_dim": bridge.act_dim})

        if mode == "bridge":
            for raw_line in sys.stdin:
                line = raw_line.strip()
                if not line:
                    continue

                response, should_exit = handle_command(bridge, json.loads(line))
                emit(response)
                if should_exit:
                    bridge = None
                    return 0
        else:
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((listen_host, listen_port))
            server.listen(1)

            with server:
                connection, _ = server.accept()
                with connection:
                    active_connection = connection
                    reader = connection.makefile("r")
                    with reader:
                        for raw_line in reader:
                            line = raw_line.strip()
                            if not line:
                                continue

                            response, should_exit = handle_command(bridge, json.loads(line))
                            send_message(connection, response)
                            if should_exit:
                                bridge = None
                                return 0
    except Exception as error:
        payload = {"status": "error", "error": str(error), "traceback": traceback.format_exc()}
        try:
            if active_connection is not None:
                send_message(active_connection, payload)
            else:
                emit(payload)
        except Exception:
            emit(payload)
        return 1
    finally:
        if bridge is not None:
            bridge.close()


if __name__ == "__main__":
    raise SystemExit(main())
