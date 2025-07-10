import strawberry
from typing import List, Optional
from datetime import datetime

@strawberry.type
class Person:
    id: str
    name: str
    role: str
    email: Optional[str]

@strawberry.type
class Visit:
    visitId: str
    visitDate: datetime
    purpose: str
    duration: int
    person: Person

@strawberry.type
class Location:
    locationId: str
    locationName: str
    capacity: int
    date: datetime
    system: str
    status: str
    hasChildren: List["Location"]
    visits: List[Visit]

    @staticmethod
    def from_dict(data: dict) -> "Location":
        return Location(
            locationId=data["locationId"],
            locationName=data["locationName"],
            capacity=data["capacity"],
            date=datetime.fromisoformat(data["date"].replace("Z", "+00:00")),
            system=data["system"],
            status=data["status"],
            hasChildren=[Location.from_dict(child) for child in data.get("hasChildren", [])],
            visits=[
                Visit(
                    visitId=v["visitId"],
                    visitDate=datetime.fromisoformat(v["visitDate"].replace("Z", "+00:00")),
                    purpose=v["purpose"],
                    duration=v["duration"],
                    person=Person(**v["person"])
                ) for v in data.get("visits", [])
            ]
        )

@strawberry.type
class Query:
    @strawberry.field
    def locations(self) -> List[Location]:
        import json
        with open("../../resources/sets/data-2.json") as f:
            raw_data = json.load(f)
        return [Location.from_dict(loc) for loc in raw_data]

schema = strawberry.Schema(Query)