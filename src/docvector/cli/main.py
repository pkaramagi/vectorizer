from pathlib import Path
import typer

from core.config.settings import settings
from core.embedders.openai_embedder import OpenAIEmbedder
from core.text_splitters.character_splitter import CharacterTextSplitter
from core.vectorizer_pipeline import VectorizationPipeline
from core.vectors.chroma_db import ChromaDB
from core.vectors.elastic_search import ElasticSearchDB
from core.embedders.huggingface_embedder import HuggingFaceEmbedder

app = typer.Typer()



@app.command()
def process(
    path:str = typer.Argument(...,help="File or Directory to process"),
    db_type:str = typer.Option("chroma",help="Vector database type[chroma|elasticsearch]"),
    embedder_type:str = typer.Option("Openai", help="Embedder type[openai|huggingface]")
):
    try:
        
        if embedder_type == "huggingface":
            embedder = HuggingFaceEmbedder()
        else: 
            embedder = OpenAIEmbedder(settings.openai.model_name)  

        if db_type == "chroma":
            
            vector_db = ChromaDB(
                settings.chroma.collection_name,
                settings.chroma.persist_dir
            )
        elif db_type == "elasticsearch":
            vector_db = ElasticSearchDB(
                settings.elasticsearch.host,
                settings.elasticsearch.port
            )
        else:
            raise NotImplementedError(f"DB type {db_type} not supported")
        
        pipeline = VectorizationPipeline(
            splitter=CharacterTextSplitter(),
            embedder=embedder,
            vector_db=vector_db
        )

        if Path(path).is_dir():
            pipeline.process_directory(path)
        else:
            pipeline.process_document(path)
    except Exception as e:
        typer.echo(e)
    
@app.command()
def query(
    question: str,
    top_k: int = typer.Option(5, help="Number of results to return"),
    db_type: str = typer.Option("chroma", help="Vector database type [chroma|elasticsearch]"),
    embedder_type:str = typer.Option("Openai", help="Embedder type[openai|huggingface]")
):
    """Query the vector database with a question"""
    try:
        if embedder_type == "huggingface":
            embedder = HuggingFaceEmbedder()
        else: 
            embedder = OpenAIEmbedder(settings.openai.model_name)
        
        # Initialize vector DB client
        if db_type == "chroma":
            vector_db = ChromaDB(
                settings.chroma.collection_name,
                settings.chroma.persist_dir
            )
        elif db_type == "elasticsearch":
            vector_db = ElasticSearchDB(
                settings.elasticsearch.host,
                settings.elasticsearch.port
            )
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
        
        
        query_embedding = embedder.embed(question)
      
        
        # Perform similarity search
        results = vector_db.search_vectors(query_embedding, top_k)
        print(results)
        # Display results
        typer.echo("\nSearch results:")
        for i, result in enumerate(results, 1):
            typer.echo(f"\nResult {i}:")
            typer.echo(f"Score: {result['score']:.4f}")
            typer.echo(f"Content: {result['document'][:200]}...")
            if result['metadata']:
                typer.echo(f"Metadata: {result['metadata']}")
                
        return results
            
    except Exception as e:
        raise typer.Exit(code=1)
    
        


if __name__ == "__main__":
    app()

