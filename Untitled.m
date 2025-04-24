data = load('Pv_Data.mat')
c=find(p==max(p));
c1=find(v==0);
c2=find(i==min(i));
a=[v(c2),i(c1),v(c),i(c),p(c)];
plot(v,i)
axis([0,140,0,140])
