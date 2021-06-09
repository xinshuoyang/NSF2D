function afun(x,y,ϵ)
    (8*E^sin(4*pi*x)^2*pi^2*cos((2*pi*y)/eps)*sin(4*pi*y)*
     -     (1.1 + sin((2*pi*x)/eps)))/(eps*(1.1 + sin((2*pi*y)/eps))^2) + 
     -  (16*E^sin(4*pi*x)^2*pi^2*cos(4*pi*x)*cos(4*pi*y)*
     -     cos((2*pi*x)/eps)*sin(4*pi*x))/(eps*(1.1 + sin((2*pi*y)/eps)))\
     -   - (16*E^sin(4*pi*x)^2*pi^2*cos(4*pi*y)*
     -     (1.1 + sin((2*pi*x)/eps)))/(1.1 + sin((2*pi*y)/eps)) + 
     -  (32*E^sin(4*pi*x)^2*pi^2*cos(4*pi*x)^2*cos(4*pi*y)*
     -     (1.1 + sin((2*pi*x)/eps)))/(1.1 + sin((2*pi*y)/eps)) - 
     -  (32*E^sin(4*pi*x)^2*pi^2*cos(4*pi*y)*sin(4*pi*x)^2*
     -     (1.1 + sin((2*pi*x)/eps)))/(1.1 + sin((2*pi*y)/eps)) + 
     -  (64*E^sin(4*pi*x)^2*pi^2*cos(4*pi*x)^2*cos(4*pi*y)*
     -     sin(4*pi*x)^2*(1.1 + sin((2*pi*x)/eps)))/
     -   (1.1 + sin((2*pi*y)/eps))
end