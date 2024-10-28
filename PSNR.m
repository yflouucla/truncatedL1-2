function p = PSNR(x,y,m)

% psnr - compute the Peak Signal to Noise Ratio, defined by :
%       PSNR(x,y) = 10*log10( m^2 / |x-y|^2 ).

d=mean((x(:)-y(:)).^2);
maxx=max(x(:));
maxy=max(y(:));
if abs(maxx-maxy)>100
    warning('The intensity ranges of the two images may be different')
end
if ~exist('m','var')
    if max(maxx,maxy)>50
        m=255;
    else
        m = 1;
    end
end
p = double(10*log10( m^2/d ));
end