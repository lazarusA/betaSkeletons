using Base.Threads, Distributed
addprocs(Sys.CPU_THREADS - 1)
@everywhere using SharedArrays
@everywhere using GR, LinearAlgebra, Random, Distributions
@everywhere function pointsDelaunay(dimBase::Float64, dim::Int64)
    points = rand(dim, 2)
    if dimBase != 1.0
        points[:,2] /= dimBase
    end
    n, tri = GR.delaunay(points[:,1], points[:,2])
    n, tri, points
end
@everywhere function triDelaunay(points)
    n, tri = GR.delaunay(points[:,1], points[:,2])
    n, tri, points
end
@everywhere function rcentersGeqOneLune(beta::Float64, x1::Float64, 
        y1::Float64, x2::Float64, y2::Float64)
    c1 = [beta.*x1./2.0 .+ (1.0 .- beta./2).*x2 beta.*y1./2.0 .+ (1.0 .- beta./2.0).*y2]
    c2 = [(1.0 .- beta./2.0).*x1 .+ beta.*x2./2.0 (1.0 .- beta./2).*y1 .+ beta.*y2./2]
    r = beta.*sqrt((x2 .- x1).^2 + (y2 .- y1).^2)./2.0
    r, c1, c2
end
@everywhere function rcentersLeqOne(beta::Float64, x1::Float64, y1::Float64,
        x2::Float64, y2::Float64)
    dot_v1 = [x1, y1]
    dot_v2 = [x2, y2]
    vec_v1v2 = dot_v2 .- dot_v1
    rotvec = reverse(vec_v1v2).*[-1,1]
    radOut = sqrt.(sum(vec_v1v2.^2))./(2*beta)
    b_out = sqrt.(1 .- beta^2)./(2*beta)
    p_c1 = (dot_v1 .+ dot_v2)./2 .+ b_out.*rotvec
    p_c2 = (dot_v1 .+ dot_v2)./2 .- b_out.*rotvec
    radOut, p_c1', p_c2'
end
@everywhere function rcentersGeqOneCircle(beta::Float64, x1::Float64, 
        y1::Float64, x2::Float64, y2::Float64)
    dot_v1 = [x1, y1]
    dot_v2 = [x2, y2]
    vec_v1v2 = dot_v2 .- dot_v1
    rotvec = reverse(vec_v1v2).*[-1,1]
    radOut = beta.*sqrt.(sum(vec_v1v2.^2))./2
    b_out = sqrt.(beta^2 .- 1)./2
    p_c1 = (dot_v1 .+ dot_v2)./2 .+ b_out.*rotvec
    p_c2 = (dot_v1 .+ dot_v2)./2 .- b_out.*rotvec
    radOut, p_c1', p_c2'
end
@everywhere function betaRandSkeletonGeqOneLune(beta::Float64, dim::Int64, 
        plotSkeleton = false, dimBase::Float64 = 1.0)
    n, tri, points = pointsDelaunay(dimBase, dim)
    mAdjacency = diagm(0 => randn(dim))
    indices = [1, 1, 2]
    j = [2, 3, 3]
    for trindx in 1:n
        for (indx, i) in enumerate(indices)
            indx1 = tri[trindx,:][i]
            indx2 = tri[trindx,:][j[indx]]
            x1, y1 = points[indx1, :]
            x2, y2 = points[indx2, :]
            r, c1, c2 = rcentersGeqOneLune(beta, x1, y1, x2, y2)
            full_1 = sqrt.(sum((points .- c1).^2, dims = 2)) .< r
            full_2 = sqrt.(sum((points .- c2).^2, dims = 2)) .< r
            full_1[[indx1, indx2]] .= false
            if any(full_1 .+ full_2 .== 2) == false
                rnd = randn()./sqrt(2)
                mAdjacency[indx1, indx2] = rnd
                mAdjacency[indx2, indx1] = rnd  
            end
        end
    end
    plotSkeleton == true ? (mAdjacency, points) : mAdjacency
end
@everywhere function betaRandSkeletonGeqOneCircle(beta::Float64, dim::Int64,
        plotSkeleton = false, dimBase::Float64 = 1.0)
    n, tri, points = pointsDelaunay(dimBase, dim)
    mAdjacency = diagm(0 => randn(dim))
    indices = [1, 1, 2]
    j = [2, 3, 3]
    for trindx in 1:n
        for (indx, i) in enumerate(indices)
            indx1 = tri[trindx,:][i]
            indx2 = tri[trindx,:][j[indx]]
            x1, y1 = points[indx1, :]
            x2, y2 = points[indx2, :]
            r, c1, c2 = rcentersGeqOneCircle(beta, x1, y1, x2, y2)
            full_1 = sqrt.(sum((points .- c1).^2, dims = 2)) .< r
            full_2 = sqrt.(sum((points .- c2).^2, dims = 2)) .< r
            full_1[[indx1, indx2]] .= false
            full_2[[indx1, indx2]] .= false
            if any(full_1 .| full_2) == false
                rnd = randn()./sqrt(2)
                mAdjacency[indx1, indx2] = rnd
                mAdjacency[indx2, indx1] = rnd  
            end
        end
    end
    plotSkeleton == true ? (mAdjacency, points, tri) : mAdjacency
end
@everywhere function betaRandSkeletonLeqOne(beta::Float64, dim::Int64, plotSkeleton = false, dimBase::Float64 = 1.0)
    mAdjacency = diagm(0 => randn(dim))
    points = rand(dim, 2)
    if dimBase != 1.0
        points[:,2] /= dimBase
    end
    for indx1 in 1:dim-1
        for indx2 in indx1 + 1:dim
            x1, y1 = points[indx1, :]
            x2, y2 = points[indx2, :]
            r, c1, c2 = rcentersLeqOne(beta, x1, y1, x2, y2)
            full_1 = sqrt.(sum((points .- c1).^2, dims = 2)) .< r
            full_2 = sqrt.(sum((points .- c2).^2, dims = 2)) .< r
            full_1[[indx1, indx2]] .= false
            if any(full_1 .+ full_2 .== 2) == false
                rnd = randn()./sqrt(2)
                mAdjacency[indx1, indx2] = rnd
                mAdjacency[indx2, indx1] = rnd  
            end
        end
    end    
    plotSkeleton == true ? (mAdjacency, points) : mAdjacency
end
@everywhere function betaSkeletonGeqOneLune(beta::Float64, points, plotSkeleton = false)
    n, tri, points = triDelaunay(points)
    dim = length(points[:,1])
    mAdjacency = diagm(0 => randn(dim))
    indices = [1, 1, 2]
    j = [2, 3, 3]
    for trindx in 1:n
        for (indx, i) in enumerate(indices)
            indx1 = tri[trindx,:][i]
            indx2 = tri[trindx,:][j[indx]]
            x1, y1 = points[indx1, :]
            x2, y2 = points[indx2, :]
            r, c1, c2 = rcentersGeqOneLune(beta, x1, y1, x2, y2)
            full_1 = sqrt.(sum((points .- c1).^2, dims = 2)) .< r
            full_2 = sqrt.(sum((points .- c2).^2, dims = 2)) .< r
            full_1[[indx1, indx2]] .= false
            if any(full_1 .+ full_2 .== 2) == false
                rnd = randn()./sqrt(2)
                mAdjacency[indx1, indx2] = rnd
                mAdjacency[indx2, indx1] = rnd  
            end
        end
    end
    plotSkeleton == true ? (mAdjacency, points) : mAdjacency
end

@everywhere function betaSkeletonGeqOneCircle(beta::Float64, points, plotSkeleton = false)
    n, tri, points = triDelaunay(points)
    dim = length(points[:,1])
    mAdjacency = diagm(0 => randn(dim))
    indices = [1, 1, 2]
    j = [2, 3, 3]
    for trindx in 1:n
        for (indx, i) in enumerate(indices)
            indx1 = tri[trindx,:][i]
            indx2 = tri[trindx,:][j[indx]]
            x1, y1 = points[indx1, :]
            x2, y2 = points[indx2, :]
            r, c1, c2 = rcentersGeqOneCircle(beta, x1, y1, x2, y2)
            full_1 = sqrt.(sum((points .- c1).^2, dims = 2)) .< r
            full_2 = sqrt.(sum((points .- c2).^2, dims = 2)) .< r
            full_1[[indx1, indx2]] .= false
            full_2[[indx1, indx2]] .= false
            if any(full_1 .| full_2) == false
                rnd = randn()./sqrt(2)
                mAdjacency[indx1, indx2] = rnd 
                mAdjacency[indx2, indx1] = rnd
            end
        end
    end
    plotSkeleton == true ? (mAdjacency, points, tri) : mAdjacency
end
@everywhere function betaSkeletonLeqOne(beta::Float64, points)
    dim = length(points[:,1])
    mAdjacency = diagm(0 => rand(dim))
    for indx1 in 1:dim-1
        for indx2 in indx1 + 1:dim
            x1, y1 = points[indx1, :]
            x2, y2 = points[indx2, :]
            r, c1, c2 = rcentersLeqOne(beta, x1, y1, x2, y2)
            full_1 = sqrt.(sum((points .- c1).^2, dims = 2)) .< r
            full_2 = sqrt.(sum((points .- c2).^2, dims = 2)) .< r
            full_1[[indx1, indx2]] .= false
            if any(full_1 .+ full_2 .== 2) == false
                rnd = randn()./sqrt(2)
                mAdjacency[indx1, indx2] = rnd
                mAdjacency[indx2, indx1] = rnd  
            end
        end
    end    
    mAdjacency
end
@everywhere function betaSkeleton(beta::Float64, dim::Int64, dimBase::Float64 = 1.0)
    if beta < 1.0
        madea, points = betaRandSkeletonLeqOne(beta, dim, true, dimBase)
        return points, madea
    elseif beta >= 1.0
        madeaLune, points =  betaRandSkeletonGeqOneLune(beta, dim, true, dimBase)
        madeaCircle = betaSkeletonGeqOneCircle(beta, points)
        return points, madeaLune, madeaCircle 
    end
end

@everywhere function betaSkeletonV2(beta::Float64, dim::Int64, dimBase::Float64 = 1.0)
    if beta < 1.0
        madea, points = betaRandSkeletonLeqOne(beta, dim, true, dimBase)
        return points, madea
    elseif beta >= 1.0
        madeaLune, points =  betaRandSkeletonGeqOneLune(beta, dim, true, dimBase)
        madeaCircle = betaSkeletonGeqOneCircle(beta, points)
        return points, madeaLune, madeaCircle
    end
end


function pointsHexagonX(center, size, i)
    angledeg = 60*i #- 30
    anglerad = pi/180*angledeg
    (center[1] .+ size*cos(anglerad), center[2] .+ size*sin(anglerad))
end

