import os
import os.path as osp
from src.utils.docker import manage_container



@manage_container
def main(*args, **kwargs):
    ...



if __name__ == '__main__':
    unstructured_container_name = "unstructured"
    unstructured_container_path = osp.join(os.getcwd(), "src", "unstructured")

    main(path = unstructured_container_path, container = unstructured_container_name)
