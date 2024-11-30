import random

def get_problem_texts():
    # random variable
    a, b, c, d, e, f, g, h = (random.choice([i for i in range(-10, 11) if i != 0]) for _ in range(8))

    # problems of diff in Korea Institute of Curriculum and Evaluation (KICE)
    problem_texts = [
        [
            f"함수 f(x) = ${a}x^3$ + $ax^2$ {'+' if b >= 0 else ''} {b}x {'+' if c >= 0 else ''}{c}\n에 대하여 f'({d}) = {e}을 만족시킬 때, \n상수 a의 값을 구하시오.",
            f"함수 f(x) = ${a}x^2$ + ax {'+' if c >= 0 else ''}{c}\n에 대하여 f'({d}) = {e}을 만족시킬 때, \n상수 a의 값을 구하시오.",
            f"함수 f(x) = ${a}x^3$ + $ax^2$ {'+' if b >= 0 else ''} {b}x {'+' if c >= 0 else ''}{c}\n에 대하여 f'({d}) = {e}일 때, \n상수 a의 값을 구하시오.",
            f"함수 f(x) = ${a}x^2$ + ax {'+' if c >= 0 else ''}{c}\n에 대하여 f'({d}) = {e}일 때, \n상수 a의 값을 구하시오.",
            f"함수 f(x) = ${a}x^3$ + $ax^2$ {'+' if b >= 0 else ''} {b}x {'+' if c >= 0 else ''}{c}\n에 대하여 f'({d}) = {e}을 만족시킬 때, \n상수 a의 값은?",
            f"함수 f(x) = ${a}x^2$ + ax {'+' if c >= 0 else ''}{c}\n에 대하여 f'({d}) = {e}을 만족시킬 때, \n상수 a의 값은?",
            f"함수 f(x) = ${a}x^3$ + $ax^2$ {'+' if b >= 0 else ''} {b}x {'+' if c >= 0 else ''}{c}\n에 대하여 f'({d}) = {e}일 때, \n상수 a의 값은?",
            f"함수 f(x) = ${a}x^2$ + ax {'+' if c >= 0 else ''}{c}\n에 대하여 f'({d}) = {e}일 때, \n상수 a의 값은?"
        ],
        [
            f"함수 f(x) = ${a}\\log_{abs(b)}({abs(c)}x)$에 대하여\n f'(x)의 값을 구하시오.",
            f"함수 f(x) = ${a}\\log_{abs(b)}({abs(c)}x)$일 때,\n f'(x)의 값을 구하시오.",
            f"함수 f(x) = ${a}\\log_{abs(b)}({abs(c)}x)$에 대하여\n f'(x)의 값은?",
            f"함수 f(x) = ${a}\\log_{abs(b)}({abs(c)}x)$일 때,\n f'(x)의 값은?"
        ],
        [
            f"매개변수 t로 나타내어진 곡선\n x = ${a}e^{{{b}t}}$ {'+' if c >= 0 else ''} {c}, y = {d}t {'+' if e >= 0 else ''} {e}\n에서 t = {f}일 때, ${{dy}}/{{dx}}$의 값은?",
            f"매개변수 t로 나타내어진 곡선\n x = ${a}e^{{{b}t}}$ {'+' if c >= 0 else ''} {c}, y = ${d}t^2$ {'+' if e >= 0 else ''} {e}t {'+' if f >= 0 else ''} {f}\n에서 t = {g}일 때, ${{dy}}/{{dx}}$의 값은?",
            f"매개변수 t로 나타내어진 곡선\n x = ${a}e^{{{b}t}}$ {'+' if c >= 0 else ''} {c}, y = {d}t {'+' if e >= 0 else ''} {e}\n에서 t = {f} 일 때, ${{dy}}/{{dx}}$의 값을 구하시오.",
            f"매개변수 t로 나타내어진 곡선\n x = ${a}e^{{{b}t}}$ {'+' if c >= 0 else ''} {c}, y = ${d}t^2$ {'+' if e >= 0 else ''} {e}t {'+' if f >= 0 else ''} {f}\n에서 t = {g}일 때, ${{dy}}/{{dx}}$의 값을 구하시오."
        ],
        [
            f"함수 f(x) = {a}$x^3$ {'+' if b >= 0 else ''} {b}$x^2$ {'+' if c >= 0 else ''} {c}x {'+' if d >= 0 else ''} {d}에 대하여 \nf'(x)의 값은?",
            f"함수 f(x) = {b}$x^2$ {'+' if c >= 0 else ''} {c}x {'+' if d >= 0 else ''} {d}에 대하여 \nf'(x)의 값은?",
            f"함수 f(x) = {a}$x^3$ {'+' if b >= 0 else ''} {b}$x^2$ {'+' if c >= 0 else ''} {c}x {'+' if d >= 0 else ''} {d}을 만족할 때, \nf'(x)의 값은?",
            f"함수 f(x) = {b}$x^2$ {'+' if c >= 0 else ''} {c}x {'+' if d >= 0 else ''} {d}을 만족할 때, \nf'(x)의 값은?",
            f"함수 f(x) = {a}$x^3$ {'+' if b >= 0 else ''} {b}$x^2$ {'+' if c >= 0 else ''} {c}x {'+' if d >= 0 else ''} {d}에 대하여 \nf'(x)의 값을 구하시오.",
            f"함수 f(x) = {b}$x^2$ {'+' if c >= 0 else ''} {c}x {'+' if d >= 0 else ''} {d}에 대하여 \nf'(x)의 값을 구하시오.",
            f"함수 f(x) = {a}$x^3$ {'+' if b >= 0 else ''} {b}$x^2$ {'+' if c >= 0 else ''} {c}x {'+' if d >= 0 else ''} {d}을 만족할 때, \nf'(x)의 값을 구하시오.",
            f"함수 f(x) = {b}$x^2$ {'+' if c >= 0 else ''} {c}x {'+' if d >= 0 else ''} {d}을 만족할 때, \nf'(x)의 값을 구하시오."
        ],
        [
            f"함수 $f(x) = {a}x^3 + ax^2 {'+' if b >= 0 else ''} {b}x {'+' if d >= 0 else ''}{d}$에 대하여 \nx = {e}에서의 미분계수가 {f}을 만족시킬 때,\n 상수 a의 값을 구하시오.",
            f"함수 $f(x) = {a}x^2 + ax {'+' if d >= 0 else ''}{d}$에 대하여 \nx = {e}에서의 미분계수가 {f}을 만족시킬 때,\n 상수 a의 값을 구하시오.",
            f"함수 $f(x) = {a}x^3 + ax^2 {'+' if b >= 0 else ''} {b}x {'+' if d >= 0 else ''}{d}$에 대하여 \nx = {e}에서의 미분계수가 {f}일 때,\n 상수 a의 값을 구하시오.",
            f"함수 $f(x) = {a}x^2 + ax {'+' if d >= 0 else ''}{d}$에 대하여 \nx = {e}에서의 미분계수가 {f}일 때,\n 상수 a의 값을 구하시오.",
            f"함수 $f(x) = {a}x^3 + ax^2 {'+' if b >= 0 else ''} {b}x {'+' if d >= 0 else ''}{d}$에 대하여 \nx = {e}에서의 미분계수가 {f}을 만족시킬 때,\n 상수 a의 값은?",
            f"함수 $f(x) = {a}x^2 + ax {'+' if d >= 0 else ''}{d}$에 대하여 \nx = {e}에서의 미분계수가 {f}을 만족시킬 때,\n 상수 a의 값은?",
            f"함수 $f(x) = {a}x^3 + ax^2 {'+' if b >= 0 else ''} {b}x {'+' if d >= 0 else ''}{d}$에 대하여 \nx = {e}에서의 미분계수가 {f}일 때,\n 상수 a의 값은?",
            f"함수 $f(x) = {a}x^2 + ax {'+' if d >= 0 else ''}{d}$에 대하여 \nx = {e}에서의 미분계수가 {f}일 때,\n 상수 a의 값은?"
        ],
        [
            f"좌표평면 위를 움직이는 점 P의 시각 t에서의 \n위치(x, y)가 x = {a}ln{b}t, y = {'' if c / d >= 0 else '-'} ${{{abs(e)}t}}/ln{abs(f)}$ 이다.\n t = $e^{abs(g)}$에서 점 P의 속력은?",
            f"좌표평면 위를 움직이는 점 P의 시각 t에서의 \n위치(x, y)가 x = {a}ln{b}t, y = {'' if c / d >= 0 else '-'} ${{{abs(e)}t}}/ln{abs(f)}$ 이다.\n t = $e^{abs(g)}$에서 점 P의 속력을 구하시오."
        ],
        [
            f"실수 전체의 집합에서 미분가능한 함수 f(x)가 \n모든 실수 x에 대하여\n f({a}$x^3$ {'+' if b >= 0 else ''} {b}$x^2$ {'+' if c >= 0 else ''} {c}x {'+' if d >= 0 else ''} {d}) = $e^{{{e}}}$\n을 만족시킬 때, f'({f})의 값을 구하시오.",
            f"실수 전체의 집합에서 미분가능한 함수 f(x)가 \n모든 실수 x에 대하여\n f({a}$x^2$ {'+' if b >= 0 else ''} {b}x {'+' if c >= 0 else ''} {c}) = $e^{{{d}}}$\n을 만족시킬 때, f'({e})의 값을 구하시오.",
            f"실수 전체의 집합에서 미분가능한 함수 f(x)가 \n모든 실수 x에 대하여\n f({a}$x^3$ {'+' if b >= 0 else ''} {b}$x^2$ {'+' if c >= 0 else ''} {c}x {'+' if d >= 0 else ''} {d}) = $e^{{{e}}}$\n을 만족시킬 때, f'({f})의 값은?",
            f"실수 전체의 집합에서 미분가능한 함수 f(x)가 \n모든 실수 x에 대하여\n f({a}$x^2$ {'+' if b >= 0 else ''} {b}x {'+' if c >= 0 else ''} {c}) = $e^{{{d}}}$\n을 만족시킬 때, f'({e})의 값은?",
            f"실수 전체의 집합에서 미분가능한 함수 f(x)가 \n모든 실수 x에 대하여\n f({a}$x^3$ {'+' if b >= 0 else ''} {b}$x^2$ {'+' if c >= 0 else ''} {c}x {'+' if d >= 0 else ''} {d}) = $e^{{{e}}}$\n일 때, f'({f})의 값을 구하시오.",
            f"실수 전체의 집합에서 미분가능한 함수 f(x)가 \n모든 실수 x에 대하여\n f({a}$x^2$ {'+' if b >= 0 else ''} {b}x {'+' if c >= 0 else ''} {c}) = $e^{{{d}}}$\n일 때, f'({e})의 값을 구하시오.",
            f"실수 전체의 집합에서 미분가능한 함수 f(x)가 \n모든 실수 x에 대하여\n f({a}$x^3$ {'+' if b >= 0 else ''} {b}$x^2$ {'+' if c >= 0 else ''} {c}x {'+' if d >= 0 else ''} {d}) = $e^{{{e}}}$\n일 때, f'({f})의 값은?",
            f"실수 전체의 집합에서 미분가능한 함수 f(x)가 \n모든 실수 x에 대하여\n f({a}$x^2$ {'+' if b >= 0 else ''} {b}x {'+' if c >= 0 else ''} {c}) = $e^{{{d}}}$\n일 때, f'({e})의 값은?"
        ]]

    return problem_texts

def maker(cls):

    # get randomly choice from each type
    problem_text = random.choice(get_problem_texts()[cls])

    return problem_text


