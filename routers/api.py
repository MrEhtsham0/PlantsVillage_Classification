from fastapi import APIRouter
from routers.v1 import model_test_router

router=APIRouter()

router.include_router(model_test_router.router)