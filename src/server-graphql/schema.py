import strawberry

from typing import List

from .data import USERS


# ============================================================
# GRAPHQL TYPES
# ============================================================

@strawberry.type
class Product:

    id: str
    name: str
    price: float


@strawberry.type
class Order:

    id: str
    total: float
    status: str
    products: List[Product]


@strawberry.type
class User:

    id: str
    name: str
    email: str
    age: int
    city: str
    country: str
    orders: List[Order]


# ============================================================
# CONVERTERS
# ============================================================

def create_product(data):

    return Product(
        id=data["id"],
        name=data["name"],
        price=data["price"]
    )


def create_order(data):

    return Order(
        id=data["id"],
        total=data["total"],
        status=data["status"],
        products=[
            create_product(product)
            for product in data["products"]
        ]
    )


def create_user(data):

    return User(
        id=data["id"],
        name=data["name"],
        email=data["email"],
        age=data["age"],
        city=data["city"],
        country=data["country"],
        orders=[
            create_order(order)
            for order in data["orders"]
        ]
    )


# ============================================================
# QUERY
# ============================================================

@strawberry.type
class Query:

    @strawberry.field
    def users(self) -> List[User]:

        return [
            create_user(user)
            for user in USERS
        ]

    @strawberry.field
    def user(self, id: str) -> User | None:

        for user in USERS:

            if user["id"] == id:
                return create_user(user)

        return None


schema = strawberry.Schema(
    query=Query
)