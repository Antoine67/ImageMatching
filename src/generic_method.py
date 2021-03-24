import abc

class GenericMethod:

    img_full = None
    img_temp = None
    
    grayscale = False
        
    
    def set_pictures(self, full, template):
        self.img_full = full
        self.img_temp = template
        
        
    @abc.abstractmethod 
    def match(self):
        """Method documentation goes there"""
        return
        