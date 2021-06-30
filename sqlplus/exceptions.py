class SQLPlusError(Exception):
    pass


class NoRowToInsert(SQLPlusError):
    "Where there's no row to write to a database"
    pass


class NoRowToWrite(SQLPlusError):
    "When there's no row to write to a CSV file"
    pass


class InvalidGroup(SQLPlusError):
    pass


class UnknownConfig(SQLPlusError):
    pass


class ReservedKeyword(SQLPlusError):
    pass


class InvalidColumns(SQLPlusError):
    pass


class TableDuplication(SQLPlusError):
    pass


class NoSuchTableFound(SQLPlusError):
    pass


class SkipThisTurn(SQLPlusError):
    pass
