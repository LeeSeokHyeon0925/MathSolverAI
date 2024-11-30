from finding_classification import *
from return_solve import *

sample_problem = '''함수 f(x) = 2x^2 + ax -3에 대하여 f'(4) = -3일 때, 상수 a의 값을 구하시오.'''  # sample_problem

prediction = classification_problem(sample_problem)
solving(prediction, sample_problem)

'''
diff-0    함수 f(x) = 2x^2 + ax -3에 대하여 f'(4) = -3일 때, 상수 a의 값을 구하시오.
diff-1    함수 f(x) = 5log_10(3x)에 대하여 f'(x)의 값은?
diff-2    매개변수 t로 나타내어진 곡선 x = -10e^6t + 2, y = -7t^2 + 3t  -3에서 t = 10일 때, dy/dx의 값을 구하시오.
diff-3    함수 f(x) = 3x^2  -7x  -3을 만족할 때, f'(x)의 값을 구하시오.
diff-4    함수 f(x) = -5x^2 + ax -5에 대하여 x = 10에서의 미분계수가 -2을 만족시킬 때, 상수 a의 값은?
diff-5    좌표평면 위를 움직이는 점 P의 시각 t에서의 위치(x, y)가 x = -3ln4t, y = - 4t/ln6 이다. t = e^4에서 점 P의 속력은?
diff-6    실수 전체의 집합에서 미분가능한 함수 f(x)가 모든 실수 x에 대하여 f(9x^3  -3x^2 + 3x  -7) = e^9일 때, f'(10)의 값은?
'''