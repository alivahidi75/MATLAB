function PlotResults(Targets,Outputs,Title)

    subplot(2,2,[1 2]);
    plot(Targets,'b');
    hold on;
    plot(Outputs,'r');
    legend('Targets','Outputs');

    if nargin>2
        title(Title);
    end
    
    Errors=Targets-Outputs;
    
    subplot(2,2,3);
    plot(Errors);

    subplot(2,2,4);
    histfit(Errors,50);

end