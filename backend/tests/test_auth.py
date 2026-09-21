"""Tests for user authentication endpoints with PostgreSQL/SQL database."""

import unittest
from fastapi.testclient import TestClient

from app import app
from core.database import SessionLocal
from models.user import User


class TestAuthAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.test_email = "tester_forge@example.com"
        self.test_password = "SecurePassword123!"
        self.test_name = "Forge Tester"

        # Clean up any leftover test user
        db = SessionLocal()
        try:
            db.query(User).filter(User.email == self.test_email).delete()
            db.commit()
        finally:
            db.close()

    def tearDown(self):
        db = SessionLocal()
        try:
            db.query(User).filter(User.email == self.test_email).delete()
            db.commit()
        finally:
            db.close()

    def test_register_and_login_flow(self):
        # 1. Register new user
        reg_res = self.client.post(
            "/api/auth/register",
            json={
                "email": self.test_email,
                "password": self.test_password,
                "full_name": self.test_name,
            },
        )
        self.assertEqual(reg_res.status_code, 200, reg_res.text)
        data = reg_res.json()
        self.assertIn("access_token", data)
        self.assertEqual(data["user"]["email"], self.test_email)
        self.assertEqual(data["user"]["full_name"], self.test_name)
        token = data["access_token"]

        # 2. Duplicate registration should fail (400)
        dup_res = self.client.post(
            "/api/auth/register",
            json={
                "email": self.test_email,
                "password": self.test_password,
            },
        )
        self.assertEqual(dup_res.status_code, 400)

        # 3. Log in with wrong password should fail (401)
        bad_login = self.client.post(
            "/api/auth/login",
            json={
                "email": self.test_email,
                "password": "WrongPassword123",
            },
        )
        self.assertEqual(bad_login.status_code, 401)

        # 4. Log in with correct credentials
        good_login = self.client.post(
            "/api/auth/login",
            json={
                "email": self.test_email,
                "password": self.test_password,
            },
        )
        self.assertEqual(good_login.status_code, 200)
        login_data = good_login.json()
        self.assertIn("access_token", login_data)

        # 5. Access /api/auth/me with valid token
        me_res = self.client.get(
            "/api/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(me_res.status_code, 200)
        me_data = me_res.json()
        self.assertEqual(me_data["email"], self.test_email)
        self.assertEqual(me_data["full_name"], self.test_name)

        # 6. Access /api/auth/me without token should fail (401)
        unauth_res = self.client.get("/api/auth/me")
        self.assertEqual(unauth_res.status_code, 401)


if __name__ == "__main__":
    unittest.main()
