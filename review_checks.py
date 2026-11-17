"""Промежуточные проверки лабораторной работы по RRT и RRT*."""

import math
import random

import kdtree
from shapely import intersects, linestrings
from shapely.geometry import Point, Polygon

from utils.node import RRTNode


def review_checks(stage, **context):
    """Выполняет проверки одного этапа, сохраняя состояние генератора random.

    Этапы:
        helpers: distance, in_goal_region, create_random_state,
            find_nearest_neighbour, extend, is_trajectory_clear, in_collision.
        rrtstar_helpers: find_near_neighbours, choose_parent, rewire.
        rrt / rrtstar: planner — соответствующая функция поиска.
        result: result, map_size, start, goal, obstacles, max_transition,
            max_iter, goal_region, algorithm ("rrt" или "rrtstar").

    Все assert-ы находятся здесь, а в ноутбуках остаются вызовы этой функции.
    Проверки результата используют math.dist и Shapely независимо от реализации
    учебных функций. Неуспех при ограниченном бюджете сам по себе не ошибка.
    """
    random_state = random.getstate()
    try:
        if stage == "helpers":
            distance = context["distance"]
            in_goal_region = context["in_goal_region"]
            sample = context["create_random_state"]
            nearest = context["find_nearest_neighbour"]
            extend = context["extend"]
            clear = context["is_trajectory_clear"]
            collision = context["in_collision"]

            for a, b in [((0, 0), (3, 4)), ((-2, 1), (2, -2)), ((1, 1), (1, 1))]:
                assert math.isclose(distance(a, b), math.dist(a, b)), "Неверное расстояние"
            assert in_goal_region((3, 4), (0, 0), 5), "Граница входит в целевую область"
            assert not in_goal_region((3, 4), (0, 0), 4.9), "Точка за пределами цели"
            assert in_goal_region((1, 2), (1, 2), 0), "Цель нулевого радиуса"

            random.seed(17)
            uniform = [sample(10, 8, 0, (5, 4), 1) for _ in range(1000)]
            assert all(0 <= x <= 10 and 0 <= y <= 8 for x, y in uniform), "Выход за карту"
            assert any(math.dist(point, (5, 4)) > 1 for point in uniform), "goal_bias=0"
            assert abs(sum(x for x, _ in uniform) / len(uniform) - 5) < 0.5, "Выборка по x"
            assert abs(sum(y for _, y in uniform) / len(uniform) - 4) < 0.4, "Выборка по y"
            for goal in [(5, 4), (0, 0), (10, 8)]:
                biased = [sample(10, 8, 1, goal, 2) for _ in range(200)]
                assert all(0 <= x <= 10 and 0 <= y <= 8 for x, y in biased), "Выход за карту"
                assert all(math.dist(point, goal) <= 2 + 1e-10 for point in biased), "Радиус выборки"
                assert len(set(biased)) > 1, "Нужна выборка в круге, а не только центр"
            assert tuple(sample(10, 8, 1, (5, 4), 0)) == (5, 4), "Нулевой радиус выборки"

            root = RRTNode((0, 0), None, 0)
            neighbour = RRTNode((3, 4), root, 5)
            assert root.children == [neighbour], "Конструктор добавляет вершину к детям родителя"
            assert neighbour.children == [], "У новой вершины ещё нет детей"
            assert root.children is not neighbour.children, "У каждой вершины свой список детей"
            tree = kdtree.create([root, neighbour])
            assert nearest(tree, (2.9, 4)) is neighbour, "Ближайший сосед"
            assert extend(tree, [], (0, 0), 2) == (False, None), "Повторная точка"
            single = kdtree.create([root])
            for target, step in [((3, 4), 2), ((0.3, 0.4), 2)]:
                success, node = extend(single, [], target, step)
                expected = min(math.dist(root.state, target), step)
                assert success and node is not None, "Свободное расширение должно пройти"
                assert node.parent is root, "Родитель новой вершины"
                assert node in root.children and node.children == [], "Дети после расширения дерева"
                assert math.isclose(math.dist(root.state, node.state), expected), "Длина шага"
                assert math.isclose(node.g, expected), "Стоимость новой вершины"
                assert len(list(single.inorder())) == 1, "extend не добавляет вершину в KD-дерево"
            wall = [[(1, 0), (2, 0), (2, 3), (1, 3)]]
            assert collision((1, 1), wall), "Касание границы считается столкновением"
            assert not clear((0, 1), (3, 1), wall), "Отрезок проходит сквозь препятствие"
            assert clear((0, 4), (3, 4), wall), "Свободный отрезок"
            children_before = list(root.children)
            assert extend(single, wall, (3, 1), 5) == (False, None), "Нельзя добавить ребро сквозь стену"
            assert root.children == children_before, "Неудачное расширение не создаёт детей"

        elif stage == "rrtstar_helpers":
            near = context["find_near_neighbours"]
            choose_parent = context["choose_parent"]
            rewire = context["rewire"]
            random.seed(90)
            nodes = [RRTNode((random.random(), random.random())) for _ in range(100)]
            nodes.extend([RRTNode((0, 0)), RRTNode((0.1, 0)), RRTNode((3, 0))])
            tree = kdtree.create(nodes)
            for state in [(0, 0), (0.3, 0.6), (1, 1)]:
                for radius in [0, 0.05, 0.1, 0.3, 1, 2, 3]:
                    expected = {node for node in nodes if math.dist(node.state, state) < radius}
                    assert set(near(tree, state, radius)) == expected, "Неверная окрестность"

            root = RRTNode((0, 0), None, 0)
            detour = RRTNode((0, 4), root, 4)
            candidate = RRTNode((3, 4), detour, 7)
            choose_parent(candidate, [root], [])
            assert candidate.parent is root and math.isclose(candidate.g, 5), "Выбор лучшего родителя"
            assert candidate not in detour.children and candidate in root.children, "Списки детей после выбора родителя"
            choose_parent(candidate, [], [])
            assert candidate.parent is root and math.isclose(candidate.g, 5), "Запасной родитель"
            choose_parent(candidate, [root], [])
            assert root.children.count(candidate) == 1, "В списке детей не должно быть повторов"

            root = RRTNode((0, 0), None, 0)
            old_parent = RRTNode((0, 5), root, 5)
            neighbour = RRTNode((4, 0), old_parent, 5 + math.sqrt(41))
            child = RRTNode((5, 0), neighbour, neighbour.g + 1)
            grandchild = RRTNode((6, 0), child, child.g + 1)
            second = RRTNode((2, 1), old_parent, 5 + math.sqrt(20))
            new_node = RRTNode((2, 0), root, 2)
            rewire(new_node, [root, new_node, neighbour, child, second], [])
            assert neighbour.parent is new_node and math.isclose(neighbour.g, 4), "Смена родителя у соседа"
            assert child.parent is neighbour and math.isclose(child.g, 5), "Обновление ребёнка"
            assert grandchild.parent is child and math.isclose(grandchild.g, 6), "Обновление внука"
            assert second.parent is new_node and math.isclose(second.g, 3), "Смена родителей у нескольких вершин"
            assert root.parent is None and new_node.parent is root, "Цикл через предка"
            assert old_parent.children == [], "Вершины удаляются из списка детей старого родителя"
            assert new_node.children == [neighbour, second], "Дети нового родителя"
            assert neighbour.children == [child] and child.children == [grandchild], "Потомки сохраняют своих родителей"
            rewire(new_node, [neighbour, child, second], [])
            assert new_node.children == [neighbour, second], "Повторный вызов не дублирует детей"

            # Родитель и его ребёнок могут оба получить нового родителя за один вызов.
            for reverse_order in (False, True):
                root = RRTNode((0, 0), None, 0)
                detour = RRTNode((0, 5), root, 5)
                neighbour = RRTNode((4, 0), detour, 5 + math.sqrt(41))
                child = RRTNode((4, 3), neighbour, neighbour.g + 3)
                grandchild = RRTNode((5, 3), child, child.g + 1)
                new_node = RRTNode((2, 0), root, 2)
                near_nodes = [child, neighbour] if reverse_order else [neighbour, child]
                rewire(new_node, near_nodes, [])
                assert neighbour.parent is new_node and child.parent is new_node, "Смена родителя у двух поколений"
                assert set(new_node.children) == {neighbour, child}, "Дети после двух изменений"
                assert neighbour.children == [] and child.children == [grandchild], "Вершина удалена из прежнего списка детей"
                assert detour.children == [], "Старый родитель больше не содержит вершину"
                assert math.isclose(child.g, 2 + math.sqrt(13)), "Стоимость после смены родителя"
                assert math.isclose(grandchild.g, child.g + 1), "Стоимость внука после двух изменений"

            wall = [[(1, 0), (2, 0), (2, 3), (1, 3)]]
            root = RRTNode((0, 1), None, 0)
            upper_left = RRTNode((0, 4), root, 3)
            upper_right = RRTNode((3, 4), upper_left, 6)
            old_parent = RRTNode((3, 3), upper_right, 7)
            candidate = RRTNode((3, 1), old_parent, 9)
            choose_parent(candidate, [root], wall)
            assert candidate.parent is old_parent and candidate.g == 9, "Выбор родителя сквозь стену"
            assert old_parent.children == [candidate] and candidate not in root.children, "Дети при отклонённом выборе родителя"
            new_node = RRTNode((0.5, 1), root, 0.5)
            rewire(new_node, [candidate], wall)
            assert candidate.parent is old_parent and candidate.g == 9, "Новое ребро проходит сквозь стену"
            assert old_parent.children == [candidate] and new_node.children == [], "Дети при отклонённой смене родителя"

        elif stage in ("rrt", "rrtstar"):
            planner = context["planner"]
            cases = [
                ((1, 1), (1, 1), 0, 0, 1, 20, True, 0),
                ((1, 1), (9, 9), 0, 0, 1, 20, False, 0),
                ((1, 1), (9, 9), 10, 0, 1, 20, True, 1 if stage == "rrt" else 10),
                ((1, 1), (9, 1), 20, 0, 1, 2, True, 4 if stage == "rrt" else 20),
                ((1, 1), (2, 1), 10, 2, 0, 2, True, 0),
            ]
            for start, goal, budget, region, bias, step, found, iterations in cases:
                random.seed(12)
                result = planner(10, 10, [], *start, *goal, max_transition=step,
                                 max_iter=budget, goal_region=region, goal_bias=bias)
                review_checks("result", result=result, map_size=(10, 10), start=start,
                              goal=goal, obstacles=[], max_transition=step,
                              max_iter=budget, goal_region=region, algorithm=stage)
                assert result[0] == found and result[2] == iterations, "Пограничный случай поиска"

            wall = [[(4, 0), (6, 0), (6, 10), (4, 10)]]
            random.seed(3)
            result = planner(10, 10, wall, 1, 5, 9, 5, max_transition=2,
                             max_iter=100, goal_region=1)
            review_checks("result", result=result, map_size=(10, 10), start=(1, 5),
                          goal=(9, 5), obstacles=wall, max_transition=2,
                          max_iter=100, goal_region=1, algorithm=stage)
            assert not result[0] and result[2] == 100, "Непроходимая стена"

            # Несколько бюджетов одного потока выборки: RRT* продолжает улучшение.
            if stage == "rrtstar":
                costs = []
                for budget in [100, 300]:
                    random.seed(42)
                    result = planner(10, 10, [], 1, 1, 9, 9, max_transition=2,
                                     max_iter=budget, goal_region=1, goal_bias=0.1)
                    review_checks("result", result=result, map_size=(10, 10), start=(1, 1),
                                  goal=(9, 9), obstacles=[], max_transition=2,
                                  max_iter=budget, goal_region=1, algorithm=stage)
                    assert result[0], "Простой тест RRT* должен найти путь"
                    costs.append(result[1].g)
                assert costs[1] <= costs[0] + 1e-8, "Лучший путь RRT* не должен ухудшаться"

        elif stage == "result":
            result = context["result"]
            start, goal = context["start"], context["goal"]
            map_w, map_h = context["map_size"]
            max_iter = context["max_iter"]
            max_transition = context["max_transition"]
            goal_region = context["goal_region"]
            algorithm = context["algorithm"]
            assert algorithm in ("rrt", "rrtstar"), "Неизвестный алгоритм"
            assert len(result) == 5, "Ожидается пятёрка результатов"
            found, end_node, iterations, tree, points = result
            assert isinstance(found, bool), "found должен быть bool"
            assert isinstance(iterations, int) and 0 <= iterations <= max_iter, "Число итераций"
            assert points, "Дерево должно содержать корень"
            nodes = {node for node, _ in points}
            assert len(nodes) == len(points), "Одна вершина добавлена несколько раз"
            root = points[0][0]
            assert tuple(root.state) == tuple(start), "Неверный корень"
            assert root.parent is None and root.g == 0, "Родитель и стоимость корня"
            added_at = [iteration for _, iteration in points]
            assert all(isinstance(i, int) for i in added_at), "Номера итераций должны быть целыми"
            assert added_at[0] == 0 and added_at == sorted(set(added_at)), "Порядок добавления"
            assert added_at[-1] <= iterations, "Вершина добавлена после завершения поиска"
            assert tree is not None, "KD-дерево возвращается и при неуспехе"
            tree_nodes = [item.data for item in tree.inorder()]
            assert len(tree_nodes) == len(points) and set(tree_nodes) == nodes, "Состав KD-дерева"

            polygons = [Polygon(obstacle) for obstacle in context["obstacles"]]
            assert not any(polygon.covers(Point(start)) for polygon in polygons), "Старт в препятствии"
            child_nodes = set()
            edges = []
            for node, _ in points:
                assert len(node.state) == 2 and all(math.isfinite(v) for v in node.state), "Координаты"
                assert 0 <= node.state[0] <= map_w and 0 <= node.state[1] <= map_h, "Вершина вне карты"
                assert math.isfinite(node.g) and node.g >= 0, "Стоимость должна быть конечной и неотрицательной"
                assert isinstance(node.children, list), "Дети хранятся в списке"
                for child in node.children:
                    assert child in nodes and child.parent is node, "Связи parent и children не согласованы"
                    assert child not in child_nodes, "Вершина повторяется в списках детей"
                    child_nodes.add(child)
                if node is root:
                    continue
                assert node.parent in nodes and node.parent is not node, "Неверный родитель"
                length = math.dist(node.parent.state, node.state)
                assert 0 < length <= max_transition + 1e-8, "Нарушено ограничение max_transition"
                assert math.isclose(node.g, node.parent.g + length, abs_tol=1e-8), "Неверная стоимость g"
                edges.append([node.parent.state, node.state])
            assert child_nodes == nodes - {root}, "Каждая вершина, кроме корня, должна быть в children своего родителя"
            if edges:
                trajectories = linestrings(edges)
                for polygon in polygons:
                    assert not intersects(trajectories, polygon).any(), "Ребро пересекает препятствие"
            seen = set()
            stack = [root]
            while stack:
                node = stack.pop()
                assert node not in seen, "Цикл в дереве"
                seen.add(node)
                stack.extend(node.children)
            assert seen == nodes, "В дереве есть цикл или вершины, не связанные с корнем"

            goal_nodes = [node for node in nodes if math.dist(node.state, goal) <= goal_region]
            assert found == bool(goal_nodes), "Флаг found не соответствует целевым вершинам"
            assert found == (end_node is not None), "Неверный end_node при неуспехе"
            if found:
                assert end_node in goal_nodes, "Конец пути вне целевой области"
                path_length = 0
                current = end_node
                while current.parent is not None:
                    path_length += math.dist(current.state, current.parent.state)
                    current = current.parent
                assert current is root and math.isclose(path_length, end_node.g, abs_tol=1e-8), "Длина пути"
            if math.dist(start, goal) <= goal_region:
                assert end_node is root and iterations == 0, "Старт уже в целевой области"
            elif algorithm == "rrtstar":
                assert iterations == max_iter, "RRT* должен использовать весь бюджет"
                if found:
                    assert math.isclose(end_node.g, min(node.g for node in goal_nodes), abs_tol=1e-8), "Выбрана не лучшая целевая вершина"
            elif found:
                assert dict(points)[end_node] == iterations, "RRT останавливается на первой целевой вершине"
            else:
                assert iterations == max_iter, "Поиск завершён до исчерпания бюджета"

        else:
            raise ValueError(f"Неизвестный этап проверки: {stage}")

        if stage != "result":
            print(f"Проверки {stage} пройдены")
    finally:
        random.setstate(random_state)
