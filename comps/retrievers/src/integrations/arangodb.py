import os
from typing import Any

import openai
from arango import ArangoClient
from arango.database import StandardDatabase
from langchain_arangodb import ArangoVector
from langchain_community.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceHubEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from comps import CustomLogger, OpeaComponent, OpeaComponentRegistry, ServiceType
from comps.cores.proto.api_protocol import RetrievalRequestArangoDB

from .config import (
    ARANGO_DB_NAME,
    ARANGO_DISTANCE_STRATEGY,
    ARANGO_GRAPH_NAME,
    ARANGO_NUM_CENTROIDS,
    ARANGO_PASSWORD,
    ARANGO_TRAVERSAL_ENABLED,
    ARANGO_TRAVERSAL_MAX_DEPTH,
    ARANGO_TRAVERSAL_MAX_RETURNED,
    ARANGO_URL,
    ARANGO_USE_APPROX_SEARCH,
    ARANGO_USERNAME,
    HUGGINGFACEHUB_API_TOKEN,
    OPENAI_API_KEY,
    OPENAI_CHAT_ENABLED,
    OPENAI_CHAT_MAX_TOKENS,
    OPENAI_CHAT_MODEL,
    OPENAI_CHAT_TEMPERATURE,
    OPENAI_EMBED_ENABLED,
    OPENAI_EMBED_MODEL,
    SUMMARIZER_ENABLED,
    TEI_EMBED_MODEL,
    TEI_EMBEDDING_ENDPOINT,
    VLLM_API_KEY,
    VLLM_ENDPOINT,
    VLLM_MAX_NEW_TOKENS,
    VLLM_MODEL_ID,
    VLLM_TEMPERATURE,
    VLLM_TIMEOUT,
    VLLM_TOP_P,
)

logger = CustomLogger("retriever_arango")
logflag = os.getenv("LOGFLAG", False)

ARANGO_TEXT_FIELD = "text"
ARANGO_EMBEDDING_FIELD = "embedding"


@OpeaComponentRegistry.register("OPEA_RETRIEVER_ARANGO")
class OpeaArangoRetriever(OpeaComponent):
    """A specialized retriever component derived from OpeaComponent for ArangoDB retriever services.

    Attributes:
        client (ArangoDB): An instance of the ArangoDB client for vector database operations.
    """

    def __init__(self, name: str, description: str, config: dict = None):
        super().__init__(name, ServiceType.RETRIEVER.name.lower(), description, config)

        self.initialize_arangodb()

        if SUMMARIZER_ENABLED:
            self.initialize_llm()

    def initialize_llm(self):
        """Initialize the language model for summarization if enabled."""
        if OPENAI_API_KEY and OPENAI_CHAT_ENABLED:
            if logflag:
                logger.info("OpenAI API Key is set. Verifying its validity...")

            openai.api_key = OPENAI_API_KEY

            try:
                openai.models.list()
                if logflag:
                    logger.info("OpenAI API Key is valid.")
                self.llm = ChatOpenAI(
                    temperature=OPENAI_CHAT_TEMPERATURE, model=OPENAI_CHAT_MODEL, max_tokens=OPENAI_CHAT_MAX_TOKENS
                )
            except openai.error.AuthenticationError:
                if logflag:
                    logger.info("OpenAI API Key is invalid.")
            except Exception as e:
                if logflag:
                    logger.info(f"An error occurred while verifying the API Key: {e}")

        elif VLLM_ENDPOINT:
            self.llm = ChatOpenAI(
                openai_api_key=VLLM_API_KEY,
                openai_api_base=f"{VLLM_ENDPOINT}/v1",
                model=VLLM_MODEL_ID,
                temperature=VLLM_TEMPERATURE,
                max_tokens=VLLM_MAX_NEW_TOKENS,
                top_p=VLLM_TOP_P,
                timeout=VLLM_TIMEOUT,
            )
        else:
            raise HTTPException(status_code=400, detail="No LLM environment variables are set, cannot generate graphs.")

    def initialize_arangodb(self):
        """Initialize the ArangoDB connection."""
        self.client = ArangoClient(hosts=ARANGO_URL)
        sys_db = self.client.db(name="_system", username=ARANGO_USERNAME, password=ARANGO_PASSWORD, verify=True)

        if not sys_db.has_database(ARANGO_DB_NAME):
            sys_db.create_database(ARANGO_DB_NAME)

        self.db = self.client.db(name=ARANGO_DB_NAME, username=ARANGO_USERNAME, password=ARANGO_PASSWORD, verify=True)
        if logflag:
            logger.info(f"Connected to ArangoDB {self.db.version()}.")

    def check_health(self) -> bool:
        """Checks the health of the retriever service."""
        if logflag:
            logger.info("[ check health ] start to check health of ArangoDB")
        try:
            version = self.db.version()
            if logflag:
                logger.info(f"[ check health ] Successfully connected to ArangoDB {version}!")
            return True
        except Exception as e:
            logger.info(f"[ check health ] Failed to connect to ArangoDB: {e}")
            return False

    def fetch_neighborhoods(
        self,
        db: StandardDatabase,
        keys: list[str],
        graph_name: str,
        search_start: str,
        query_embedding: list[float],
        collection_name: str,
        traversal_max_depth: int,
        traversal_max_returned: int,
    ) -> dict[str, Any]:
        """Fetch the neighborhoods of matched documents"""

        if traversal_max_depth < 1:
            traversal_max_depth = 1
        
        if traversal_max_returned < 1:
            traversal_max_returned = 1

        sub_query = ""
        neighborhoods = {}

        if search_start == "chunk":
            sub_query = f"""
                FOR node IN 1..1 INBOUND doc {graph_name}_HAS_SOURCE
                    FOR node2, edge IN 1..{traversal_max_depth} ANY node {graph_name}_LINKS_TO
                        LET score = COSINE_SIMILARITY(edge.{ARANGO_EMBEDDING_FIELD}, @query_embedding)
                        SORT score DESC
                        LIMIT {traversal_max_returned}
                        RETURN edge.{ARANGO_TEXT_FIELD}
            """

        elif search_start == "edge":
            sub_query = f"""
                FOR chunk IN {graph_name}_SOURCE
                    FILTER chunk._key == doc.source_id
                    LIMIT 1
                    RETURN chunk.{ARANGO_TEXT_FIELD}
            """

        elif search_start == "node":
            sub_query = f"""
                FOR node, edge IN 1..{traversal_max_depth} ANY doc {graph_name}_LINKS_TO
                    LET score = COSINE_SIMILARITY(edge.{ARANGO_EMBEDDING_FIELD}, @query_embedding)
                    SORT score DESC
                    LIMIT {traversal_max_returned}

                    FOR chunk IN {graph_name}_SOURCE
                        FILTER chunk._key == edge.source_id
                        LIMIT 1
                        RETURN {{[edge.{ARANGO_TEXT_FIELD}]: chunk.{ARANGO_TEXT_FIELD}}}
            """

        query = f"""
            FOR doc IN @@collection
                FILTER doc._key IN @keys

                LET neighborhood = (
                    {sub_query}
                )

                RETURN {{[doc._key]: neighborhood}}
        """

        bind_vars = {
            "@collection": collection_name,
            "query_embedding": query_embedding,
            "keys": keys,
        }

        cursor = db.aql.execute(query, bind_vars=bind_vars)

        for doc in cursor:
            neighborhoods.update(doc)

        return neighborhoods

    def generate_prompt(self, query: str, text: str) -> str:
        """Generate a prompt for summarization."""
        return f"""
            I've performed vector similarity on the following
            query to retrieve most relevant documents: '{query}' 

            Each retrieved Document may have a 'RELATED INFORMATION' section.

            Please consider summarizing the Document below using the query as the foundation to summarize the text.

            The Document: {text}

            Provide a summary to include all content relevant to the query, using the RELATED INFORMATION section (if provided) as needed.

            Your summary:
        """

    async def invoke(self, input: RetrievalRequestArangoDB) -> list:
        """Process the retrieval request and return relevant documents."""
        if logflag:
            logger.info(input)

        #################
        # Process Input #
        #################

        query = input.input
        embedding = input.embedding if isinstance(input.embedding, list) else None
        graph_name = input.graph_name or ARANGO_GRAPH_NAME
        search_start = input.search_start
        enable_traversal = input.enable_traversal or ARANGO_TRAVERSAL_ENABLED
        enable_summarizer = input.enable_summarizer or SUMMARIZER_ENABLED
        distance_strategy = input.distance_strategy or ARANGO_DISTANCE_STRATEGY
        use_approx_search = input.use_approx_search or ARANGO_USE_APPROX_SEARCH
        num_centroids = input.num_centroids or ARANGO_NUM_CENTROIDS
        traversal_max_depth = input.traversal_max_depth or ARANGO_TRAVERSAL_MAX_DEPTH
        traversal_max_returned = input.traversal_max_returned or ARANGO_TRAVERSAL_MAX_RETURNED

        search_start = input.search_start
        if search_start == "node":
            collection_name = f"{graph_name}_ENTITY"
        elif search_start == "edge":
            collection_name = f"{graph_name}_LINKS_TO"
        elif search_start == "chunk":
            collection_name = f"{graph_name}_SOURCE"
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid search_start value: {search_start}. Expected 'node', 'edge', or 'chunk'.",
            )

        if logflag:
            logger.info(f"Graph name: {graph_name}, Start Collection name: {collection_name}")

        #################
        # Validate Data #
        #################

        if not self.db.has_graph(graph_name):
            if logflag:
                graph_names = [g["name"] for g in self.db.graphs()]
                logger.error(f"Graph '{graph_name}' does not exist in ArangoDB. Graphs: {graph_names}")
            return []

        if not self.db.graph(graph_name).has_vertex_collection(collection_name):
            if logflag:
                collection_names = self.db.graph(graph_name).vertex_collections()
                m = f"Collection '{collection_name}' does not exist in graph '{graph_name}'. Collections: {collection_names}"
                logger.error(m)
            return []

        collection = self.db.collection(collection_name)
        collection_count = collection.count()

        if collection_count == 0:
            if logflag:
                logger.error(f"Collection '{collection_name}' is empty.")
            return []

        if collection_count < num_centroids:
            if logflag:
                m = f"Collection '{collection_name}' has fewer documents ({collection_count}) than the number of centroids ({num_centroids}). Please adjust the number of centroids."
                logger.error(m)
            return []

        ################################
        # Retrieve Embedding Dimension #
        ################################

        random_doc = collection.random()
        random_doc_id = random_doc["_id"]
        embedding = random_doc.get(ARANGO_EMBEDDING_FIELD)

        if not embedding:
            if logflag:
                logger.error(f"Document '{random_doc_id}' is missing field '{ARANGO_EMBEDDING_FIELD}'.")
            return []

        if not isinstance(embedding, list):
            if logflag:
                logger.error(f"Document '{random_doc_id}' has a non-list embedding field, found {type(embedding)}.")
            return []

        dimension = len(embedding)

        if dimension == 0:
            if logflag:
                logger.error(f"Document '{random_doc_id}' has an empty embedding field.")
            return []

        if OPENAI_API_KEY and OPENAI_EMBED_MODEL and OPENAI_EMBED_ENABLED:
            embeddings = OpenAIEmbeddings(model=OPENAI_EMBED_MODEL, dimensions=dimension)
        elif TEI_EMBEDDING_ENDPOINT and HUGGINGFACEHUB_API_TOKEN:
            embeddings = HuggingFaceHubEmbeddings(
                model=TEI_EMBEDDING_ENDPOINT, huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
            )
        else:
            embeddings = HuggingFaceBgeEmbeddings(model_name=TEI_EMBED_MODEL)

        vector_db = ArangoVector(
            embedding=embeddings,
            embedding_dimension=dimension,
            database=self.db,
            collection_name=collection_name,
            embedding_field=ARANGO_EMBEDDING_FIELD,
            text_field=ARANGO_TEXT_FIELD,
            distance_strategy=distance_strategy,
            num_centroids=num_centroids,
        )

        ######################
        # Compute Similarity #
        ######################

        try:
            if input.search_type == "similarity_score_threshold":
                docs_and_similarities = await vector_db.asimilarity_search_with_relevance_scores(
                    query=query,
                    embedding=embedding,
                    k=input.k,
                    score_threshold=input.score_threshold,
                    use_approx=use_approx_search,
                )
                search_res = [doc for doc, _ in docs_and_similarities]
            elif input.search_type == "mmr":
                search_res = await vector_db.amax_marginal_relevance_search(
                    query=query,
                    embedding=embedding,
                    k=input.k,
                    fetch_k=input.fetch_k,
                    lambda_mult=input.lambda_mult,
                    use_approx=use_approx_search,
                )
            else:
                search_res = await vector_db.asimilarity_search(
                    query=query,
                    embedding=embedding,
                    k=input.k,
                    use_approx=use_approx_search,
                )
        except Exception as e:
            if logflag:
                logger.error(f"Error during similarity search: {e}")
            return []

        if not search_res:
            if logflag:
                logger.info("No documents found.")
            return []

        if logflag:
            logger.info(f"Found {len(search_res)} documents.")

        ########################################
        # Traverse Source Documents (optional) #
        ########################################

        if enable_traversal:
            keys = [r.id for r in search_res]

            neighborhoods = self.fetch_neighborhoods(
                db=vector_db.db,
                keys=keys,
                graph_name=graph_name,
                search_start=search_start,
                query_embedding=embedding,
                collection_name=collection_name,
                traversal_max_depth=traversal_max_depth,
                traversal_max_returned=traversal_max_returned,
            )

            for r in search_res:
                neighborhood = neighborhoods.get(r.id)
                if neighborhood:
                    r.page_content += "\n------\nRELATED INFORMATION:\n------\n"
                    r.page_content += str(neighborhood)

            if logflag:
                logger.info(f"Added neighborhoods to {len(search_res)} documents.")

        ################################
        # Summarize Results (optional) #
        ################################

        if enable_summarizer:
            for r in search_res:
                prompt = self.generate_prompt(query, r.page_content)
                res = self.llm.invoke(prompt)
                summarized_text = res.content

                if logflag:
                    logger.info(f"Summarized {r.id}")

                r.page_content = summarized_text

        return search_res
