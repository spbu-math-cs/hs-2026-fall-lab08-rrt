class RRTNode:
    """
    Узел поиска для алгоритма RRT.

    Args
    --------
    state :
        Состояние в пространстве, соответствующее текущему узлу
    parent : RRTNode
        Указатель на родительский узел.
    g : float | int
        Длина пути до текущего узла.

    Attributes
    ----------
    children : list[RRTNode]
        Непосредственные дети узла. При создании вершина автоматически
        добавляется в список детей своего родителя, если он задан.
    """
    def __init__(self, 
                 state,
                 parent: "RRTNode" = None,
                 g: float = 0.0):
        self.state = state
        self.parent = parent
        self.g = g
        self.children = []
        if parent is not None:
            parent.children.append(self)

    def __len__(self):
        """
        Вспомогательный метод, необходимый для корректной работы KD-дерева
        """
        return len(self.state)

    def __getitem__(self, i):
        """
        Вспомогательный метод, необходимый для корректной работы KD-дерева
        """
        return self.state[i]

    def __repr__(self):
        return 'Vertex({}, {})'.format(self.state[0], self.state[1])
