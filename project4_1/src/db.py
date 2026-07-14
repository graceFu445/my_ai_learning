import sqlite3
from pathlib import Path
from typing import Optional


SAMPLE_ORDERS = [
    (
        "12345",
        "user_001",
        "shipped",
        "Wireless Headphones",
        "Arrived at Beijing Sorting Center",
        "2026-07-10T10:00:00",
    ),
    (
        "67890",
        "user_001",
        "pending_payment",
        "Smart Watch",
        "Waiting for payment",
        "2026-07-10T10:05:00",
    ),
    (
        "11223",
        "user_002",
        "delivered",
        "Laptop Stand",
        "Delivered to locker",
        "2026-07-10T10:10:00",
    ),
]


def initialize_database(db_path: Path) -> None:
    """创建本地订单库，并写入稳定的演示订单数据。"""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY,
                user_id TEXT,
                status TEXT,
                items TEXT,
                logistics_info TEXT,
                created_at TEXT
            )
            """
        )
        count = conn.execute("SELECT count(*) FROM orders").fetchone()[0]
        if count == 0:
            conn.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?)", SAMPLE_ORDERS)


def get_order(db_path: Path, order_id: str) -> Optional[dict[str, str]]:
    """根据订单号返回一条订单；订单不存在时返回 None。"""
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT order_id, user_id, status, items, logistics_info, created_at
            FROM orders
            WHERE order_id = ?
            """,
            (order_id,),
        ).fetchone()
    return dict(row) if row else None
