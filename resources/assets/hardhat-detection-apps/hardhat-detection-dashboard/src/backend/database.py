import logging
from sqlalchemy import create_engine, event
from sqlalchemy.orm import declarative_base, sessionmaker

# Database Configuration
DATABASE_URL = "sqlite:///./device_monitoring.db"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sqlalchemy.engine")

# Create Engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False,
    pool_size=10,
    max_overflow=20,
    pool_timeout=30,
    pool_recycle=3600
)

# Add a custom logger to log all SQL statements
@event.listens_for(engine, "before_cursor_execute")
def log_sql_calls(conn, cursor, statement, parameters, context, executemany):
    logger.info("Executing SQL: %s", statement)
    if parameters:
        logger.info("With parameters: %s", parameters)

# Session and Base
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
