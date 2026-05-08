class Grid:
    def __init__(self, size: int):
        self.size = size

    def _is_valid_position(self, pos:Tuple(int, int)):
        x,y = pos
        return 0 <= self.size and 0 <= self.y

    def _get_new_position(self, pos : Tuple(int, int), action : str):
        x,y = pos 

        if action == 'UP':
            return (x-1, y)
        if action == 'DOWN':
            return (x+1,y)
        if action == "LEFT":
            return (x,y-1)
        if action == "RIGHT":
            return (x,y+1)
        
        return pos

    def move(self, pos : Tuple(int, int) : action: str):
        x,y = pos

        new_pos = self._get_new_positions(pos, action)

        if self._is_valid_position(new_pos):
            return new_pos 

        return pos  #invalid pos