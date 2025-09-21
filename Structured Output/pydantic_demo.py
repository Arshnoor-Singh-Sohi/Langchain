from pydantic import BaseModel, EmailStr, Field
from typing import Optional


class Student(BaseModel):
    name: str = 'Unknown'
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(ge=0.0, le=4.0, default=0.0, description="CGPA must be between 0.0 and 4.0")


# new_student = {'name': 'John Doe'}
new_student = {}


student = Student(**new_student)

print(type(student))
print(dict(student))
print(student.name)

student_dict = student.dict()

print(student_dict['age'])

student_json = student.model_dump_json()