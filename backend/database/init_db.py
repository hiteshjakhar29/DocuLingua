import logging
import os
import sys

# Ensure backend module can be discovered cleanly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backend.database.database import engine, Base
from backend.database.models import Document, Extraction, Entity

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("InitDB")

def init_db():
    logger.info("Initializing database connection...")
    logger.info("Creating all missing table definitions schemas locally against engine metadata...")
    Base.metadata.create_all(bind=engine)
    logger.info("Successfully bound Document, Extraction, and Entity architectures out to persistence runtime.")

if __name__ == "__main__":
    init_db()
