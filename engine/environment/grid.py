import pygame 

class Grid:
    def __init__(self, grid_size : int, cell_size = 60):
        pygame.init()

        self.grid_size=grid_size
        self.cell_size=cell_size

        window_size = self.grid_size * self.cell_size
        self.screen = pygame.display.set_mode((window_size, window_size))
        pygame.display.set_caption("Raid MARL Training")

        self.colours={
            "Tank":(50,150,255),
            "Dealer":(50,255,50),
            "Healer":(255,255,50),
            "Boss":(255,50,50),
        }
        self.font = pygame.font.SysFont(None, 24)

    def render(self,gamestate):
        self.screen.fill((30,30,30))

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, (50, 50, 50), rect, 1)
        
        for agent_id in gamestate.hp.keys():
            if gamestate.is_alive(agent_id):
                pos = gamestate.positions[agent_id]
                role = gamestate.identities[agent_id].role
                color = self.colours.get(role, (255, 255, 255))
            
                center_x = int(pos[0] * self.cell_size + self.cell_size / 2)
                center_y = int(pos[1] * self.cell_size + self.cell_size / 2)

                pygame.draw.circle(self.screen, color, (center_x, center_y), self.cell_size // 3)

                text = self.font.render(role[0], True, (0, 0, 0))
                self.screen.blit(text, (center_x - 6, center_y - 8))

                hp_ratio = gamestate.hp[agent_id] / gamestate.identities[agent_id].stats.max_hp

                bar_width = self.cell_size * 0.8
                bar_rect = pygame.Rect(center_x - bar_width/2, center_y - self.cell_size/2 + 5, bar_width * hp_ratio, 6)
                pygame.draw.rect(self.screen, (0, 255, 0) if hp_ratio > 0.4 else (255, 0, 0), bar_rect)

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()