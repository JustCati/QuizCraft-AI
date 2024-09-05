import os
import docker
import subprocess
import os.path as osp



def isContainerRunning(container_name):
    client = docker.from_env()
    try:
        client.containers.get(container_name).status
    except docker.errors.NotFound:
        return False
    return True



def startContainer(path, container_name):
    if not isContainerRunning(container_name):
        subprocess.run(["docker", "compose", "-f", osp.join(path, "docker-compose.yml"), "up", "-d", container_name])
    else:
        print(f"Container {container_name} is already running!")



def stopContainer(path, container_name):
    if isContainerRunning(container_name):
        subprocess.run(["docker", "compose", "-f", osp.join(path, "docker-compose.yml"), "down", container_name])
    else:
        print(f"Container {container_name} is not running!")



def manage_container(func, *args, **kwargs):
    def wrapper(*args, **kwargs):
        path = kwargs.get("path", None)
        container_name = kwargs.get("container", None)
        if path and container_name:
            startContainer(osp.join(os.getcwd(), "src", "unstructured"), container_name=container_name)
            func(*args, **kwargs)
            stopContainer(osp.join(os.getcwd(), "src", "unstructured"), "unstructured")
        else:
            raise ValueError("Path and container name must be provided!")
    return wrapper
