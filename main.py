from stdmanage import student
students = []

def user_input():
    sym_no = input("Enter student symbol number: ")
    name = input("Enter student name: ")
    grade = input("Enter student grade: ")
    percentage = float(input("Enter student percentage: "))
    
    student_add=student(sym_no, name, grade, percentage)
    students.append(student_add)


while True:
    user_input()
    a=input("Do you want to add another student? (yes/no): ")
    if a.lower() != 'yes':
        print("Exiting the program.")
        break
    elif a.lower() == 'yes':
        continue  
    else:
        break
 
 
a=input("Do you want to see the detail of specific student (yes/no): ")
if a.lower() == 'yes':
    sym_no = input("Enter student symbol number: ")
    for student in students:
        if student.sym_no == sym_no:
            student.display()
            break
        else:
            print("Student not found.")



