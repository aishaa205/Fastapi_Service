import aio_pika
import threading
import json
import httpx
from dotenv import load_dotenv
import os
import traceback

load_dotenv()

RABBITMQ_URL = os.getenv("RABBITMQ_URL")
FASTAPI_URL = os.getenv("FASTAPI_URL")

async def send_request_to_route(endpoint: str, request_type: str, data: dict):
    """Send an HTTP request to FastAPI route dynamically."""
    url = f"{FASTAPI_URL}/{endpoint}"

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            if request_type.lower() == "post":
                response = await client.post(url, json=data)
            elif request_type.lower() == "get":
                response = await client.get(url, params=data)
            elif request_type.lower() == "put":
                response = await client.get(url, json=data)
            else:
                raise ValueError(f"Unsupported HTTP method: {request_type}")
            return response
        except httpx.RequestError as e:
            print(f"Error making request to {url}: {e}")
            # traceback.print_exc()
            return None
        except Exception as e:
            print(f"Error in consuming queue: {repr(e)}")
            # traceback.print_exc()
            return None

async def consume_queue(queue_name: str):
    conn = await aio_pika.connect_robust(
        RABBITMQ_URL,
        timeout=10,
        heartbeat=30,
        client_properties={"connection_name": "fastapi_consumer"},
    )
    channel = await conn.channel()
    queue   = await channel.declare_queue(queue_name, durable=True)

    async with queue.iterator() as qiter:
        async for message in qiter:
            async with message.process():
                try:
                    print(f"[{queue_name}] message:", message.body)
                    data         = json.loads(message.body)
                    endpoint     = data.pop("endpoint")
                    request_type = data.pop("type")
                except Exception as e:
                    print(f"[{queue_name}] bad message:", e)
                    continue

                print(f"[{queue_name}] forwarding {request_type} → {endpoint}")
                try:
                    resp = await send_request_to_route(endpoint, request_type, data)
                    # print(f"[{queue_name}] got {resp.status_code}: {await resp.text()}")
                    # await message.ack()
                except Exception as exc:
                    print(f"[{queue_name}] Error: {exc}")
                    try:
                        await message.reject(requeue=True)
                    except Exception as e:
                        print(f"[{queue_name}] Failed to reject message: {e}")