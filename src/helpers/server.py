import json
import strawberry
from fastapi import FastAPI
from strawberry.fastapi import GraphQLRouter
import uvicorn
import nest_asyncio

# Needed to allow uvicorn to run inside Jupyter
nest_asyncio.apply()

# Load your JSON data
with open("../../resources/sets/data-1.json") as f:
    people_data = json.load(f)

@strawberry.type
class Person:
    id: int
    name: str
    age: int

@strawberry.type
class Query:
    @strawberry.field
    def all_people(self) -> list[Person]:
        return [Person(**person) for person in people_data]

# Setup GraphQL
schema = strawberry.Schema(query=Query)
graphql_app = GraphQLRouter(schema)

# Create FastAPI app
app = FastAPI()
app.include_router(graphql_app, prefix="/graphql")

# Run the server inline
uvicorn.run(app, host="127.0.0.1", port=8000)

# ../resources/sets/data-1.json