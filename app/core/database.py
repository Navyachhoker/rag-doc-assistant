
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

from app.config import get_settings

#engine connection
settings = get_settings()
engine = create_engine(settings.database_url, pool_pre_ping=True)
#pool_pre_ping checks the whether the connection is still alive, 
# if dead, a fresh connection is created 
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

EMBEDDING_DIM =  3072 # gemini embedding 001 output dimension


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True)
    filename = Column(String(512), nullable=False)
    file_type = Column(String(32), nullable=False)
    # New stages replace the old binary processing/ready/failed:
    # extracting -> embedding_text -> captioning_images -> ready
    # 'failed' can happen at any stage; resume picks up from wherever it stopped.
    status = Column(String(32), default="extracting")
    uploaded_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


    #define relationship bet db tables
    #back_populates connect both tables <->
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")
    images = relationship("DocumentImage", back_populates="document", cascade="all, delete-orphan")


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    content = Column(Text, nullable=False)
    page_number = Column(Integer, nullable=True)
    chunk_index = Column(Integer, nullable=False)
    embedding = Column(Vector(EMBEDDING_DIM), nullable=True)  # NULL = not yet embedded

    document = relationship("Document", back_populates="chunks")

class DocumentImage(Base):
    __tablename__ = "document_images"

    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    file_path = Column(String(1024), nullable=False)
    caption = Column(Text, nullable=True)       # NULL = not yet captioned
    page_number = Column(Integer, nullable=True)
    embedding = Column(Vector(EMBEDDING_DIM), nullable=True)  # NULL = not yet embedded

    document = relationship("Document", back_populates="images")


def init_db() -> None:
    #Creates the vector extension + all tables. Call once on startup
    with engine.connect() as conn:
        conn.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS vector")
        conn.commit()#saves the extension creation
    Base.metadata.create_all(bind=engine)#creates all table


def get_db() -> Session:
    #FastAPI dependency — yields a session and guarantees closure
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()