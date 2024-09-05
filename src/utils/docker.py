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

