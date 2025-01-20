import logging

import psycopg2
import psycopg2.extras

from pgvector.psycopg2 import register_vector

import configparser

from utils import get_config, TColor


class DBConnector:
    def __init__(self, config: str, use_vector: bool = False):
        config: configparser.ConfigParser = get_config(config)

        logging.debug("Connecting to database")

        self.conn = psycopg2.connect(
            database=config["DB"].get("database"),
            host=config["DB"].get("host"),
            user=config["DB"].get("user"),
            password=config["DB"].get("password"),
            port=config["DB"].get("port"))

        if not self.conn:
            raise Exception("Could not connect to database.")

        self.conn.autocommit = False

        if use_vector:
            register_vector(self.conn)

    def get_cursor(self) -> psycopg2.extensions.cursor:
        return self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)


if __name__ == '__main__':
    DBConnector(config='../config.ini')
