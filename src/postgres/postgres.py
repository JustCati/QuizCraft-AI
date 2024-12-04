import os
import dotenv
import time

import psycopg2 as pg
from python_on_whales import DockerClient
from psycopg2.extensions import connection, cursor



class Postgres:
    def __init__(self, compose_file: str = "docker-compose.yml", env_file: str = "") -> None:
        self.compose_file: str = compose_file
        self.env: dict[str, str] = self.__get_env(env_file)
        self.conn, self.cursor = self.__connect()

     
    def __connect(self) -> tuple[connection, cursor]:
        self.__manage_istance("launch")
        time.sleep(5)

        if not self.env["POSTGRES_DB"] or not self.env["POSTGRES_USER"] or not self.env["POSTGRES_PASSWORD"]:
            self.__manage_istance("stop")
            raise ValueError("Database environment variables are not set")

        conn: connection = pg.connect(
            database = self.env.pop("POSTGRES_DB", ""),
            user = self.env.pop("POSTGRES_USER", ""),
            password = self.env.pop("POSTGRES_PASSWORD", ""),
            host = self.env.pop("POSTGRES_HOST", "localhost"),
            port = self.env.pop("POSTGRES_PORT", "5432"),
        )
        cursor = conn.cursor()
        return conn, cursor


    def __get_env(self, path: str = "") -> dict[str, str]:
        dotenv.load_dotenv(path)
        toRet: dict[str, str] = {
            "POSTGRES_USER": os.getenv("POSTGRES_USER"),
            "POSTGRES_PASSWORD": os.getenv("POSTGRES_PASSWORD"),
            "POSTGRES_DB": os.getenv("POSTGRES_DB"),
        }
        if not all(toRet.values()):
            raise ValueError("Database environment variables are not set")
        return toRet

    def __enter__(self) -> "Postgres":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.cursor.close()
        self.conn.close()
        self.__manage_istance("stop")


    def __manage_istance(self, action: str) -> None:
        docker: DockerClient = DockerClient(compose_files=[self.compose_file])

        running_containers: list[str] = [container.name for container in docker.compose.ps()]
        if any(["postgres" in name for name in running_containers]):
            if action == "launch":
                print("Postgres already running.")
            elif action == "stop":
                docker.compose.down(quiet=True)
                print("Postgres stopped.")
        else:
            if action == "launch":
                docker.compose.up(detach=True, wait=True, quiet=True)
                print("Postgres launched.")
            elif action == "stop":
                print("Postgres not running.")


    def launch(self) -> None:
        self.__manage_istance("launch")


    def stop(self):
        self.__manage_istance("stop")


    def does_file_exist(self, file_id: str) -> bool:
        self.cursor.execute("SELECT filename FROM document WHERE id = %s", (file_id,))
        result = self.cursor.fetchone()
        return result is not None


    def save_file_to_db(self, file_id: str, file_name: str) -> None:
        self.cursor.execute("INSERT INTO document (id, filename) VALUES (%s, %s)", (file_id, file_name))
        self.conn.commit()
