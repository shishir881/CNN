class student:
    def __init__(self,sym_no,name,grade,percentage):
        self.sym_no= sym_no
        self.name= name
        self.grade= grade
        self.percentage= percentage
    
    def display(self):
        print(self.sym_no)
        print(self.name)
        print(self.grade)
        print(self.percentage)