import boto3

def inspect():
    session = boto3.Session(profile_name="aiia", region_name="ap-southeast-2")
    ddb = session.resource("dynamodb")
    table = ddb.Table("RagDoll-Knowledge-test")
    res = table.query(KeyConditionExpression="PK = :pk", ExpressionAttributeValues={":pk": "TITLE#up-front-core"})
    items = res.get("Items", [])
    while "LastEvaluatedKey" in res:
        res = table.query(KeyConditionExpression="PK = :pk", ExpressionAttributeValues={":pk": "TITLE#up-front-core"}, ExclusiveStartKey=res["LastEvaluatedKey"])
        items.extend(res.get("Items", []))

    for term in ["radio", "killed", "sl"]:
        matching = [i for i in items if term in i.get("verbatimText", "").lower() or term in i.get("title", "").lower()]
        print(f"Term '{term}': {len(matching)} rules")
        for r in matching[:5]:
            print(f"  Rule {r.get('ruleNumber')}: {r.get('verbatimText')[:70]}...")
        print("-" * 50)

if __name__ == "__main__":
    inspect()
