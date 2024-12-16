function axicon_Pattern = FLFM_axiconTransmittance(Camera, Resolution)

% Compute lens transmittance function for one micro-lens (consitent with Advanced Optics Theory book and Broxton's paper)
axicon_Pattern = zeros( length(Resolution.yMLspace), length(Resolution.xMLspace) );
    for a=1:length(Resolution.yMLspace)
        for b=1:length(Resolution.xMLspace)
            x1 = Resolution.yMLspace(a);
            x2 = Resolution.xMLspace(b);
            r = sqrt(x1^2 + x2^2);
            k=2*pi/Camera.WaveLength;
            axicon_Pattern(a,b) = exp(-1i.*k.*(Camera.n_axicon-1).*Camera.theta.*r);
        end
    end

% Mask the pattern, to avoid overlaping when applying it to the whole image 
patternML_single = axicon_Pattern;
% circ lens shapes with with blocked light between them
[x,y] = meshgrid(Resolution.yMLspace, Resolution.xMLspace);
patternML_single((sqrt(x.*x+y.*y) >= Camera.lensPitch/2)) = 0; 
axicon_Pattern(:,:) = patternML_single;