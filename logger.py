import sqlite3
import datetime
from time import sleep
from luma.luma_calculator import PicLumaValues

DB_PATH = "./logs.db"


class DBLogger:
    def __init__(self) -> None:

        self.entrance_passes = "EntrancePasses"
        self.luma = "Luma"

        self.connection = sqlite3.connect(DB_PATH)
        self.cursor = self.connection.cursor()

        self.__create_db_if_not_exist(self.entrance_passes)
        self.__create_db_if_not_exist(self.luma)

    def log_pass(self, entered: int, left: int, inside: int) -> bool:
        timestamp = datetime.datetime.now()
        self.cursor.execute("INSERT INTO " + self.entrance_passes + " (datetime, entered, left, inside) VALUES (?, ?, ?, ?)",
                            (timestamp, entered, left, inside))
        self.connection.commit()
        return True

    # def log_luma(self, luma: float, perc_lightness: float) -> bool:
    #     timestamp = datetime.datetime.now()
    #     self.cursor.execute("INSERT INTO " + self.luma + " (datetime, luma, perc_lightness) VALUES (?, ?, ?)",
    #                         (timestamp, luma, perc_lightness))
    #     self.connection.commit()
    #     return True
    
    def log_luma(self, luma_values: PicLumaValues) -> bool:
        timestamp = datetime.datetime.now()
        values = (timestamp, ) + luma_values.as_tuple()
        self.cursor.execute("INSERT INTO " + self.luma + " (datetime,\
                            mean_luma, \
                            geom_mean_luma, \
                            mean_lightness, \
                            geom_mean_lightness, \
                            median_lightness, \
                            mean_filtered_lightness, \
                            geom_mean_filtered_lightness, \
                            median_filtered_lightness) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", values)
        self.connection.commit()
        return True
    def show_logs_pass(self):
        self.cursor.execute("SELECT * FROM " + self.entrance_passes)
        result = self.cursor.fetchall()
        for row in result:
            print(row)

    def show_logs_luma(self):
        self.cursor.execute("SELECT * FROM " + self.luma)
        result = self.cursor.fetchall()
        for row in result:
            print(row)

    def __create_db_if_not_exist(self, name: str) -> None:
        if name == self.entrance_passes:
            self.cursor.execute(
                "CREATE TABLE IF NOT EXISTS " + name + " (datetime timestamp, entered int, left int, inside int)")
        elif name == self.luma:
            self.cursor.execute( \
                "CREATE TABLE IF NOT EXISTS " + name + " ( \
                        datetime timestamp KEY, \
                        mean_luma real NOT NULL, \
                        geom_mean_luma real  NOT NULL, \
                        mean_lightness real  NOT NULL, \
                        geom_mean_lightness real NOT NULL,\
                        median_lightness real NOT NULL, \
                        mean_filtered_lightness real NOT NULL, \
                        geom_mean_filtered_lightness real NOT NULL, \
                        median_filtered_lightness real NOT NULL)")
        else:
            raise ValueError(name)
        self.connection.commit()

    def _db_cleanup(self) -> None:
        self.cursor.execute("DELETE FROM " + self.entrance_passes)
        self.cursor.execute("DELETE FROM " + self.luma)
        self.connection.commit()
    
    def drop(self) -> None:
        self.cursor.execute("DROP TABLE " + self.entrance_passes)
        self.cursor.execute("DROP TABLE " + self.luma)
        self.connection.commit()

    def __del__(self) -> None:
        self.cursor.close()
        self.connection.close()


Logger = DBLogger

if __name__ == "__main__":
    logger = Logger()

    logger.show_logs_pass()
    logger.show_logs_luma()
    
    # logger._db_cleanup()
