# Step 1: Unpack the element ---------------------------------------------
using Statistics
using FFTW
using LoopVectorization

const XI  = 1
const PXI = 2
const YI  = 3
const PYI = 4
const ZI  = 5
const PZI = 6

using SIMD
"""
Claude attempt at a complex SIMD:
"""
struct CVec{N,T}
    re::Vec{N,T}
    im::Vec{N,T}
end

struct CVec{V}
    re::V
    im::V
end

Base.:+(a::CVec, b::CVec) = CVec(a.re + b.re, a.im + b.im)
Base.:-(a::CVec, b::CVec) = CVec(a.re - b.re, a.im - b.im)
Base.:-(a::CVec) = CVec(-a.re, -a.im)
Base.:*(a::CVec, b::CVec) =
    CVec(muladd(a.re, b.re, -(a.im * b.im)),
         muladd(a.re, b.im,  a.im * b.re))
Base.:*(a::Real, b::CVec) = CVec(a * b.re, a * b.im)
Base.:*(a::CVec, b::Real) = CVec(a.re * b, a.im * b)
Base.:*(a::CVec{V}, b::V) where {V} = CVec(a.re * b, a.im * b)
Base.:*(a::V, b::CVec{V}) where {V} = CVec(a * b.re, a * b.im)
Base.:+(a::CVec{V}, b::V) where {V} = CVec(a.re + b, a.im)
Base.:+(a::V, b::CVec{V}) where {V} = CVec(a + b.re, b.im)
Base.:+(a::Real, b::CVec{V}) where {V} = CVec(a + b.re, b.im)
Base.:+(a::CVec{V}, b::Real) where {V} = CVec(a.re + b, a.im)
Base.:-(a::CVec{V}, b::V) where {V} = CVec(a.re - b, a.im)
Base.:-(a::V, b::CVec{V}) where {V} = CVec(a - b.re, -b.im)
Base.:-(a::Real, b::CVec{V}) where {V} = CVec(a - b.re, -b.im)
Base.:-(a::CVec{V}, b::Real) where {V} = CVec(a.re - b, a.im)

Base.:/(a::CVec{V}, b::V) where {V} = CVec(a.re / b, a.im / b)

Base.conj(a::CVec)        = CVec(a.re, -a.im)

@inline function cexp(z::CVec)
    m = exp(z.re)
    CVec(m * cos(z.im), m * sin(z.im))
end

# zero(::CVec) / zero(::Type{CVec{V}}) for both cases:
Base.zero(::CVec{V}) where {V} = CVec(zero(V), zero(V))
Base.zero(::Type{CVec{V}}) where {V} = CVec(zero(V), zero(V))

# vifelse: SIMD.jl defines vifelse(::Vec{N,Bool}, ::Vec, ::Vec);
# add the scalar (Bool) version so the same call works in both paths:
SIMD.vifelse(m::Vec{N,Bool}, a::CVec{V}, b::CVec{V}) where {N,V} =
    CVec(vifelse(m, a.re, b.re), vifelse(m, a.im, b.im))
SIMD.vifelse(m::Bool, a::CVec{T}, b::CVec{T}) where {T<:Real} =
    m ? a : b

@inline function Base.inv(z::CVec)
    d = muladd(z.re, z.re, z.im * z.im)
    CVec(z.re / d, -z.im / d)
end

@inline im_mul(z::CVec) = CVec(-z.im, z.re)

function vmap!(f, outre::Vector{T}, outim::Vector{T},
               xs::NTuple{K,Tuple{Vector{T},Vector{T}}}; N::Int = 4) where {T,K}
    n = length(outre)
    V = Vec{N,T}
    i = 1
    @inbounds while i + N - 1 <= n
        vs = ntuple(k -> CVec(vload(V, xs[k][1], i), vload(V, xs[k][2], i)), K)
        r  = f(vs...)
        vstore(r.re, outre, i); vstore(r.im, outim, i)
        i += N
    end
    @inbounds while i <= n
        vs = ntuple(k -> CVec(xs[k][1][i], xs[k][2][i]), K)   # scalar CVec now!
        r  = f(vs...)                                          # same `f`, e.g. fw
        outre[i], outim[i] = r.re, r.im
        i += 1
    end
    return outre, outim
end



@inline faddeeva(z) = faddeeva(z, Val(48))

@inline function __faddeeva_coeff(n, type::Type=Float64)
  N = n
  M = 2*N
  L = sqrt(N/sqrt(type(2)))
  theta_arr = [k*type(pi)/M for k in (-M+1):(M-1)]
  t_arr = L*tan.(theta_arr/2)
  f_arr = [(L^2 + t^2)*exp(-t^2) for t in t_arr]
  f_arr = vcat(zero(type), f_arr)
  a_arr = real.(fft(fftshift(f_arr)))/(2*M)
  return a_arr[2:(N+1)], L
end
@generated function faddeeva(z::CVec{V}, ::Val{M}) where {V,M}
    T = V <: Vec ? eltype(V) : V
    a_arr, L = __faddeeva_coeff(M, T)
    a_arr = reverse(a_arr)
    ex = :($(a_arr[1]))
    for i in 2:M
        ex = :(muladd($ex, Z, $(a_arr[i])))
    end
    return quote
        mask = z.im >= zero($V)
        c1   = vifelse(mask, one($V), zero($V))
        c2   = vifelse(mask, one($V), -one($V))
        z = c2*z
        lmiz_inv =inv($L - im_mul(z))
        Z = ($L + im_mul(z))*lmiz_inv
        return c1*2*cexp(-(z*z)) + c2*(2*$ex*lmiz_inv + $(1/sqrt(T(pi))))*lmiz_inv
    end
end

@inline function gaus_space_charge_kick!(i, coords::Coords,means,sigmas,L,N,locXI,locYI,locPXI,locPYI)
    #println(means)
    v = coords.v
    
    X =   v[i,locXI]  - means[locXI]
    Y =   v[i,locYI]  - means[locYI]
    Z =   v[i,ZI]  - means[ZI] 
    #PX =  v[i,PXI] - means[PXI]
    #PY =  v[i,PYI] - means[PYI]
    PZ =  v[i,PZI]
    
    sig_x = sigmas[locXI,locXI]
    sig_y = sigmas[locYI,locYI]
    sig_z = sigmas[ZI,ZI]
    sig_x = sqrt(abs(sig_x))
    sig_y = sqrt(abs(sig_y))
    sig_z = sqrt(abs(sig_z))
    r_sq = X*X + Y*Y
    r = sqrt(r_sq)
    #precompute this later? Depends on particle...
    gamma = sqrt(5.11e5*5.11e5 + means[PZI]*means[PZI])/5.11e5

    beta = sqrt(1 - 1/(gamma*gamma))#this REALLY wants precomputation, or at least taylor expansion
    #assume none are moving backwards in lab frame
    
    #Differential Q/z - oth order assuming infinite cylinder at z position
    #Consider mathamaticaing everything properly
    z_dens = N*exp(-Z*Z/(2*sig_z*sig_z))/(sig_z*sqrt(2*pi))
    #precompute
    #Why are these negative sometimes????
    sig_sq = sig_x*sig_x + sig_y*sig_y

    sig_x2 = sig_x*sig_x
    sig_y2 = sig_y*sig_y
    
    k = 1.44e-9 #ev-m

    #These can't be complex by construction
    sig_denom = sig_x*sig_x-sig_y*sig_y
    A = k*sqrt(2*pi/sig_denom)
    B = faddeeva(CVec(X/sqrt(2*sig_denom), Y/sqrt(2*sig_denom)))

    C = faddeeva(CVec(X*sig_y/sig_x/sqrt(2*sig_denom),Y*sig_x/sig_y/sqrt(2*sig_denom)))
    
    expD = exp(-X*X/(2*sig_x2) - Y*Y/(2*sig_y2))
    #force
    result = z_dens*A*(B - expD*C)

    proper_time = L/(beta*gamma)
    #impulse
    result = result*proper_time

    loc_x_impulse = result.im
    loc_y_impulse = result.re

    v[i,locPXI] = v[i,locPXI] + loc_x_impulse
    v[i,locPYI] = v[i,locPYI] + loc_y_impulse
end
@inline function gaus_space_charge(kc, bunch, L)
  backend = get_backend(bunch.coords.v)
  means, sigmas = mean_and_cov(bunch.coords.v, bunch.coords.weight, backend)
  #sigmas = Symmetric(sigmas)
  #println(sigmas[:,1])
  #println(sigmas)
  N = length(bunch.coords.v)/6
  locXI = XI
  locYI = YI
  locPXI = PXI
  locPYI = PYI
  #outsigmas = zeros(6)
  #for i in 1:6
  #      outsigmas = sigmas[i,i]
  #end
  if (sigmas[XI,XI] < sigmas[YI,YI])
        locXI = YI
        locYI = XI
        locPXI = PYI
        locPYI = PXI
  end
  #print(N)
  return push(kc, make_kernel_call(gaus_space_charge_kick!,(means,sigmas,L,N,locXI,locYI,locPXI,locPYI)))
end

# Step 1: Unpack the element ---------------------------------------------
function _track!(
  coords::Coords,
  bunch::Bunch,
  ele::LineElement, 
  context::Context,
  p_over_q_ref,
  tm,
  scalar_params,
  ramp_particle_energy_without_rf,
  ramp_update_each_particle;
  kwargs...
)
  # Unpack the line element (type unstable)
  L = float(ele.L) # Automatically calls deval (element-level get)
  # float call is required because L is allowed to be any type
  # in order to keep binaries smaller for tracking routines, 
  # we don't want to compile separate routines for Int64
  ap = deval(ele.AlignmentParams, context)
  bp = deval(ele.BendParams, context)
  bm = deval(ele.BMultipoleParams, context)
  pp = deval(ele.PatchParams, context)
  dp = deval(ele.ApertureParams, context)
  mp = deval(ele.MapParams, context)
  rp = deval(ele.RFParams, context)
  lp = deval(ele.BeamlineParams, context)
  fpp = deval(ele.FourPotentialParams, context)

  if scalar_params
    L = scalarize(L)
    ap = scalarize(ap)
    bp = scalarize(bp)
    bm = scalarize(bm)
    pp = scalarize(pp)
    dp = scalarize(dp)
    mp = scalarize(mp)
    rp = scalarize(rp)
    lp = scalarize(lp)
    fpp = scalarize(fpp)
    p_over_q_ref = scalarize(p_over_q_ref)
  end

  # Function barrier
  universal!(coords, tm, ele, ramp_particle_energy_without_rf, ramp_update_each_particle, bunch, L, p_over_q_ref, ap, bp, bm, pp, dp, rp, lp, mp, fpp; kwargs...)
end

# Step 2: Push particles through -----------------------------------------
function universal!(
  coords,
  tm,
  ele,
  ramp_particle_energy_without_rf, 
  ramp_update_each_particle,
  bunch,
  L, 
  p_over_q_ref,
  alignmentparams,
  bendparams,
  bmultipoleparams,
  patchparams,
  apertureparams,
  rfparams,
  beamlineparams,
  mapparams,
  fourpotentialparams;
  kwargs...
) 
  # Compute information about reference coordinate system:
  t_enter = bunch.t_ref
  beta_gamma_enter_t = R_to_beta_gamma(bunch.species, p_over_q_ref)
  beta_gamma_enter = p_over_q_ref isa TimeDependentParam ? beta_gamma_enter_t(t_enter) : beta_gamma_enter_t
  g = isnothing(bendparams) ? (0,0) : reverse((bendparams.g_ref .* sincos(bendparams.tilt_ref)))
  ds_step = (L == 0 || isactive(patchparams)) ? L : BeamTracking.find_steps(tm, L)[2]
  # Reference time evolution thru element assumes constant energy
  # using the energy at the start of the element:
  t_exit = bunch.t_ref + L / beta_gamma_to_v(beta_gamma_enter)
  beta_gamma_exit_t = R_to_beta_gamma(bunch.species, p_over_q_ref)

  if p_over_q_ref isa TimeDependentParam
    if ramp_update_each_particle
      beta_gamma_exit = beta_gamma_exit_t(t_exit)
      p_over_q_ref_exit = p_over_q_ref(t_exit)
    else
      beta_gamma_exit = beta_gamma_enter # Don't ramp at end of element if !ramp_update_each_particle
      p_over_q_ref_exit = p_over_q_ref(t_enter)
    end
  else
    beta_gamma_exit = beta_gamma_exit_t
    p_over_q_ref_exit = p_over_q_ref
  end

  # Current KernelChain length is 10 because we have up to
  # 2 aperture, 2 alignment, 1 body kernel, 1 IBS kernel,
  # 2 kernels to update the particles' reference energy,
  # and 2 for coordinate conversion with implicit
  kc = KernelChain(Val{11}(), RefState(; t_enter, beta_gamma_enter, t_exit, beta_gamma_exit, L, g, ds_step))

  kc = gaus_space_charge(kc, bunch, L)
    
  ramp_per_particle = p_over_q_ref isa TimeDependentParam && ramp_update_each_particle
  bunch_beta_gamma = R_to_beta_gamma(bunch.species, bunch.p_over_q_ref)

  if ramp_per_particle
    kc = push(kc, make_kernel_call(BeamTracking.reference_momentum_shift!, (bunch_beta_gamma, beta_gamma_enter_t-bunch_beta_gamma, Val{!ramp_particle_energy_without_rf}())))
    kc = push_transforms_out(kc, make_kernel_call((i, coords, cur_s, cur_t_ref)->error("transforms_out! not supported yet with ramp_update_each_particle = true")))
    kc = push_transforms_in(kc, make_kernel_call((i, coords, cur_s, cur_t_ref)->error("transforms_in! not supported yet with ramp_update_each_particle = true")))
  else
    # Make sure to evaluate p_over_q_ref if not ramp_update_each_particle
    p_over_q_ref = p_over_q_ref isa TimeDependentParam ? p_over_q_ref(t_enter) : p_over_q_ref
    if !(beta_gamma_enter ≈ bunch_beta_gamma)
        kc = push(kc, make_kernel_call(BeamTracking.reference_momentum_shift!, (bunch_beta_gamma, beta_gamma_enter - bunch_beta_gamma, Val{!ramp_particle_energy_without_rf}())))
        bunch.p_over_q_ref = p_over_q_ref
    end
  end

  # Entrance aperture and alignment
  if isactive(alignmentparams)
    if isactive(apertureparams)
      if apertureparams.aperture_shifts_with_body
        kc = @inline(alignment(tm, kc, p_over_q_ref, bunch, alignmentparams, bendparams, L, true))
        kc = @inline(aperture(tm, kc, p_over_q_ref, bunch, apertureparams, true))
      else
        kc = @inline(aperture(tm, kc, p_over_q_ref, bunch, apertureparams, true))
        kc = @inline(alignment(tm, kc, p_over_q_ref, bunch, alignmentparams, bendparams, L, true))
      end
    else
      kc = @inline(alignment(tm, kc, p_over_q_ref, bunch, alignmentparams, bendparams, L, true))
    end
  elseif isactive(apertureparams)
    kc = @inline(aperture(tm, kc, p_over_q_ref, bunch, apertureparams, true))
  end

  if ((hasfield(typeof(tm), :ibs_damping_on) && hasfield(typeof(tm), :ibs_fluctuations_on)) 
    && (tm.ibs_damping_on || tm.ibs_fluctuations_on) && L > 0)
    bp = ifelse(isactive(bendparams), bendparams, nothing)
    kc = @inline(ibs_kick(tm, kc, p_over_q_ref, bunch, bp, L))
  end

  if isactive(mapparams)    
    if isactive(bendparams)
      error("Tracking through a LineElement containing both MapParams and BendParams not currently defined")
    elseif isactive(bmultipoleparams)
      error("Tracking through a LineElement containing both MapParams and BMultipoleParams not currently defined")
    elseif isactive(rfparams)
      error("Tracking through a LineElement containing both MapParams and RFParams not currently defined")
    elseif isactive(patchparams)
      error("Tracking through a LineElement containing both MapParams and PatchParams not currently defined")
    elseif isactive(fourpotentialparams)
      error("Tracking through a LineElement containing both MapParams and FourPotentialParams not currently defined")
    else
      kc = @inline(pure_map(tm, kc, p_over_q_ref, bunch, mapparams, L))
    end

  elseif isactive(fourpotentialparams)    
    if isactive(bmultipoleparams)
      error("Tracking through a LineElement containing both FourPotentialParams and BMultipoleParams not currently defined")
    elseif isactive(rfparams)
      error("Tracking through a LineElement containing both FourPotentialParams and RFParams not currently defined")
    elseif isactive(patchparams)
      error("Tracking through a LineElement containing both MapParams and PatchParams not currently defined")
    else
      kc = @inline(implicit(tm, kc, p_over_q_ref, bunch, fourpotentialparams, bendparams, L))
    end

  elseif isactive(patchparams)    
    if isactive(alignmentparams)
      error("Tracking through a LineElement containing both PatchParams and AlignmentParams is undefined")
    elseif isactive(bendparams)
      error("Tracking through a LineElement containing both PatchParams and BendParams not currently defined")
    elseif isactive(bmultipoleparams)
      error("Tracking through a LineElement containing both PatchParams and BMultipoleParams not currently defined")
    elseif isactive(rfparams)
      error("Tracking through a LineElement containing both PatchParams and RFParams not currently defined")
    else
      # Pure patch
      kc = @inline(pure_patch(tm, kc, p_over_q_ref, bunch, patchparams, L))
    end

  elseif isactive(rfparams)
    if isactive(bendparams)
      error("Tracking through a LineElement containing both RFParams and BendParams not currently defined")
    end
    !rfparams.is_crabcavity || error("Crab cavities not yet supported for tracking")

    kc = @inline(rfcavity(tm, kc, p_over_q_ref, bunch, bmultipoleparams, rfparams, beamlineparams, L))
    
  elseif isactive(bendparams)
    if bendparams.edge1_int != 0 || bendparams.edge2_int != 0; error("edge1_int and edge2_int not yet handled for tracking"); end
    # Bend
    if !isactive(bmultipoleparams) 
      # Bend no field
      kc = @inline(bend_no_field(tm, kc, p_over_q_ref, bunch, bendparams, L))
    else
      n_multipoles = get_n_multipoles(bmultipoleparams)
      if 0 in bmultipoleparams.order # Bend-solenoid
        if n_multipoles == 1
          bm0 = first(bmultipoleparams)
          # Pure bend-solenoid
          kc = @inline(bend_pure_bsolenoid(tm, kc, p_over_q_ref, bunch, bendparams, bm0, L))
        else
          # Bend-solenoid with other multipoles of order > 0
          kc = @inline(bend_bsolenoid(tm, kc, p_over_q_ref, bunch, bendparams, bmultipoleparams, L))
        end
      elseif 1 in bmultipoleparams.order # Bend-dipole
        if n_multipoles == 1
          bm1 = first(bmultipoleparams)
          # Pure bend-dipole
          kc = @inline(bend_pure_bdipole(tm, kc, p_over_q_ref, bunch, bendparams, bm1, L))
        else
          # Bend-dipole with other multipoles of order > 1
          kc = @inline(bend_bdipole(tm, kc, p_over_q_ref, bunch, bendparams, bmultipoleparams, L))
        end
      elseif 2 in bmultipoleparams.order # Bend-quadrupole
        if n_multipoles == 1
          bm2 = first(bmultipoleparams)
          # Pure bend-quadrupole
          kc = @inline(bend_pure_bquadrupole(tm, kc, p_over_q_ref, bunch, bendparams, bm2, L))
        else
          # Bend-quadrupole with other multipoles of order > 1
          kc = @inline(bend_bquadrupole(tm, kc, p_over_q_ref, bunch, bendparams, bmultipoleparams, L))
        end
      else # Bend-multipole
        if n_multipoles == 1
          bmk = first(bmultipoleparams)
          # Pure bend-multipole
          kc = @inline(bend_pure_bmultipole(tm, kc, p_over_q_ref, bunch, bendparams, bmk, L))
        else
          # Bend-multipole with other multipoles of order > 2
          kc = @inline(bend_bmultipole(tm, kc, p_over_q_ref, bunch, bendparams, bmultipoleparams, L))
        end
      end
    end

  elseif isactive(bmultipoleparams)
    # BMultipole
    n_multipoles = get_n_multipoles(bmultipoleparams)
    if 0 in bmultipoleparams.order # Solenoid
      if n_multipoles == 1
        # Pure solenoid
        bm0 = first(bmultipoleparams)
        kc = @inline(pure_bsolenoid(tm, kc, p_over_q_ref, bunch, bm0, L))
      else
        # Solenoid with other multipoles of order > 0
        kc = @inline(bsolenoid(tm, kc, p_over_q_ref, bunch, bmultipoleparams, L))
      end
    elseif 1 in bmultipoleparams.order # Dipole without bend
      if n_multipoles == 1
        # Pure dipole
        bm1 = first(bmultipoleparams)
        kc = @inline(pure_bdipole(tm, kc, p_over_q_ref, bunch, bm1, L))
      else
        # Dipole with other multipoles of order > 1
        kc = @inline(bdipole(tm, kc, p_over_q_ref, bunch, bmultipoleparams, L))
      end
    elseif 2 in bmultipoleparams.order # Quadrupole
      if n_multipoles == 1
        # Pure quadrupole
        bm2 = first(bmultipoleparams)
        kc = @inline(pure_bquadrupole(tm, kc, p_over_q_ref, bunch, bm2, L))
      else
        # Quadrupole with other multipoles of order > 1
        kc = @inline(bquadrupole(tm, kc, p_over_q_ref, bunch, bmultipoleparams, L))
      end
    else # Higher order multipole
      if n_multipoles == 1
        # Pure multipole
        bmk = first(bmultipoleparams)
        kc = @inline(pure_bmultipole(tm, kc, p_over_q_ref, bunch, bmk, L))
      else
        # Multipole with other multipoles of order > 2
        kc = @inline(bmultipole(tm, kc, p_over_q_ref, bunch, bmultipoleparams, L))
      end
    end

  elseif L != 0
    kc = @inline(drift(tm, kc, p_over_q_ref, bunch, L))
  end

  # Exit aperture and alignment
  if isactive(alignmentparams)
    if isactive(apertureparams)
      if apertureparams.aperture_shifts_with_body
        kc = @inline(aperture(tm, kc, p_over_q_ref, bunch, apertureparams, false))
        kc = @inline(alignment(tm, kc, p_over_q_ref, bunch, alignmentparams, bendparams, L, false))
      else
        kc = @inline(alignment(tm, kc, p_over_q_ref, bunch, alignmentparams, bendparams, L, false))
        kc = @inline(aperture(tm, kc, p_over_q_ref, bunch, apertureparams, false))
      end
    else
      kc = @inline(alignment(tm, kc, p_over_q_ref, bunch, alignmentparams, bendparams, L, false))
    end
  elseif isactive(apertureparams)
    kc = @inline(aperture(tm, kc, p_over_q_ref, bunch, apertureparams, false))
  end

  # If ramping, now need to uniformly ramp all particles to same reference energy
  if ramp_per_particle
    # If TimeDependentParam, at end ramp all 
    # uniformly to p_over_q_ref(t_ref at end)
    kc = push(kc, make_kernel_call(BeamTracking.reference_momentum_shift!, (beta_gamma_enter_t, beta_gamma_exit-beta_gamma_enter_t, Val{!ramp_particle_energy_without_rf}())))
  end

  # noinline necessary here for small binaries and faster execution
  @noinline launch!(coords, kc; kwargs...)

  # Update reference time
  bunch.t_ref = t_exit
  bunch.p_over_q_ref = p_over_q_ref_exit

  return nothing
end

#---------------------------------------------------------------------------------------------------
# universal! for SaganCavity tracking.

function universal!(coords, tm::SaganCavity, ele, ramp_particle_energy_without_rf, ramp_update_each_particle, bunch, L,
  p_over_q_ref, alignmentparams, bendparams, bmultipoleparams, patchparams, apertureparams,
  rfparams, beamlineparams, mapparams, fourpotentialparams; kwargs...) 

  !isactive(mapparams) || error("SaganCavity Tracking through element $ele_name with MapParams is undefined")
  !isactive(patchparams) || error("SaganCavity Tracking through element $ele_name with PatchParams is undefined")
  !isactive(patchparams) || error("SaganCavity Tracking through element $ele_name with BendParams is undefined")
  !isactive(fourpotentialparams) || error("SaganCavity Tracking through element $ele_name with FourPotentialParams is undefined")
  isactive(rfparams) || error("SaganCavity Tracking through element $ele_name without RFParams is undefined")

  beta_gamma_ref = R_to_beta_gamma(bunch.species, bunch.p_over_q_ref)

  # Compute information about reference coordinate system:
  t_enter = bunch.t_ref
  if p_over_q_ref isa TimeDependentParam
    beta_gamma_enter = R_to_beta_gamma(bunch.species, p_over_q_ref(t_enter))
  else
    beta_gamma_enter = R_to_beta_gamma(bunch.species, p_over_q_ref)
  end
  g = isnothing(bendparams) ? (0,0) : reverse((bendparams.g_ref .* sincos(bendparams.tilt_ref)))
  ds_step = (L == 0 || isactive(patchparams)) ? L : BeamTracking.find_steps(tm, L)[2]
  # reference time change
  if L != 0
    species = bunch.species
    p1_over_q_ref = p_over_q_ref
    rf_omega = rf_omega_calc(rfparams, beamlineparams)
    n_cells, L_active = rf_step_calc(tm.n_cells, tm.L_active, rf_omega, L)
    L_outer = (L - L_active) / 2
    E1_ref = R_to_E(species, p1_over_q_ref)
    dE_ref = beamlineparams.dE_ref
    E0_ref = E1_ref - dE_ref
    dt_ref = L_outer/E_to_v(species, E0_ref) + L_outer/E_to_v(species, E1_ref)
 
    if n_cells == 0
      L_inner = L_active / 2
      dt_ref += L_inner/E_to_v(species, E0_ref) + L_inner/E_to_v(species, E1_ref)
    else
      for i_step = 1:n_cells
        E_now_ref = E0_ref + (i_step - 1/2) * dE_ref / n_cells
        dt_ref += L_active / (n_cells * E_to_v(species, E_now_ref))
      end
    end
    t_exit = dt_ref
  else
    t_exit = 0
  end
  if p_over_q_ref isa TimeDependentParam
    beta_gamma_exit = R_to_beta_gamma(bunch.species, p_over_q_ref(t_exit))
  else
    beta_gamma_exit = R_to_beta_gamma(bunch.species, p_over_q_ref)
  end
  kc = KernelChain(Val{10}(), RefState(; t_enter, beta_gamma_enter, t_exit, beta_gamma_exit, L, g, ds_step))

  # Ramping
  if p_over_q_ref isa TimeDependentParam
    if ramp_update_each_particle
      error("ramp_update_each_particle = true not yet implemented for SaganCavity") # TODO
    end
    p_over_q_ref_initial = bunch.p_over_q_ref
    p_over_q_ref_final = p_over_q_ref(bunch.t_ref)
    if !(p_over_q_ref_initial ≈ p_over_q_ref_final)
      kc = push(kc, make_kernel_call(BeamTracking.reference_momentum_shift!, (p_over_q_ref_initial, 
                                       p_over_q_ref_final-p_over_q_ref_initial, Val{!ramp_particle_energy_without_rf}())))
      setfield!(bunch, :p_over_q_ref, p_over_q_ref_final)
    end
  end

  # Entrance aperture and alignment
  if isactive(alignmentparams)
    if isactive(apertureparams)
      if apertureparams.aperture_shifts_with_body
        kc = @inline(alignment(tm, kc, p_over_q_ref, bunch, alignmentparams, bendparams, L, true))
        kc = @inline(aperture(tm, kc, p_over_q_ref, bunch, apertureparams, true))
      else
        kc = @inline(aperture(tm, kc, p_over_q_ref, bunch, apertureparams, true))
        kc = @inline(alignment(tm, kc, p_over_q_ref, bunch, alignmentparams, bendparams, L, true))
      end
    else
      kc = @inline(alignment(tm, kc, p_over_q_ref, bunch, alignmentparams, bendparams, L, true))
    end
  elseif isactive(apertureparams)
    kc = @inline(aperture(tm, kc, p_over_q_ref, bunch, apertureparams, true))
  end

  # Cavity tracking
  kc = @inline(sagan_cavity(tm, kc, p_over_q_ref, bunch, ele.name, bmultipoleparams, rfparams, beamlineparams, L))

  # Exit aperture and alignment
  if isactive(alignmentparams)
    if isactive(apertureparams)
      if apertureparams.aperture_shifts_with_body
        kc = @inline(aperture(tm, kc, p_over_q_ref, bunch, apertureparams, false))
        kc = @inline(alignment(tm, kc, p_over_q_ref, bunch, alignmentparams, bendparams, L, false))
      else
        kc = @inline(alignment(tm, kc, p_over_q_ref, bunch, alignmentparams, bendparams, L, false))
        kc = @inline(aperture(tm, kc, p_over_q_ref, bunch, apertureparams, false))
      end
    else
      kc = @inline(alignment(tm, kc, p_over_q_ref, bunch, alignmentparams, bendparams, L, false))
    end
  elseif isactive(apertureparams)
    kc = @inline(aperture(tm, kc, p_over_q_ref, bunch, apertureparams, false))
  end

  # noinline necessary here for small binaries and faster execution
  @noinline launch!(coords, kc; kwargs...)
  
  bunch.t_ref = t_exit
  
  return nothing
end

#---------------------------------------------------------------------------------------------------

# === Drift === #
@inline drift(tm, kc, p_over_q_ref, bunch, L) = error("Undefined for tracking method $tm")

# == Implicit === #
@inline implicit(tm, kc, p_over_q_ref, bunch, fourpotentialparams, bendparams, L) = error("Undefined for tracking method $tm")

# === Straight Elements === #
# "Pure" means only ONE SINGLE multipole.
# When "pure" is not present, it means that at least one HIGHER ORDER
# multipole exists.
@inline thin_pure_rf(tm, kc, p_over_q_ref, bunch, rfparams)                          = error("Undefined for tracking method $tm")
@inline thin_pure_bsolenoid(tm, kc, p_over_q_ref, bunch, bm0)                        = error("Undefined for tracking method $tm")
@inline thin_bsolenoid(tm, kc, p_over_q_ref, bunch, bmultipoleparams)                = error("Undefined for tracking method $tm")
@inline thin_pure_bdipole(tm, kc, p_over_q_ref, bunch, bm1)                          = error("Undefined for tracking method $tm")
@inline thin_bdipole(tm, kc, p_over_q_ref, bunch, bmultipoleparams)                  = error("Undefined for tracking method $tm")
@inline thin_pure_bquadrupole(tm, kc, p_over_q_ref, bunch, bm2)                      = error("Undefined for tracking method $tm")
@inline thin_bquadrupole(tm, kc, p_over_q_ref, bunch, bmultipoleparams)              = error("Undefined for tracking method $tm")
@inline thin_pure_bmultipole(tm, kc, p_over_q_ref, bunch, bmk)                       = error("Undefined for tracking method $tm")
@inline thin_bmultipole(tm, kc, p_over_q_ref, bunch, bmultipoleparams)               = error("Undefined for tracking method $tm")
@inline thin_bmultipole_rf(tm, kc, p_over_q_ref, bunch, bmultipoleparams, rfparams)  = error("Undefined for tracking method $tm")

@inline thick_pure_rf(tm, kc, p_over_q_ref, bunch, rfparams, beamlineparams, L)                         = error("Undefined for tracking method $tm")
@inline thick_pure_bsolenoid(tm, kc, p_over_q_ref, bunch, bm0, L)                                       = error("Undefined for tracking method $tm")
@inline thick_bsolenoid(tm, kc, p_over_q_ref, bunch, bmultipoleparams, L)                               = error("Undefined for tracking method $tm")
@inline thick_pure_bdipole(tm, kc, p_over_q_ref, bunch, bm1, L)                                         = error("Undefined for tracking method $tm")
@inline thick_bdipole(tm, kc, p_over_q_ref, bunch, bmultipoleparams, L)                                 = error("Undefined for tracking method $tm")
@inline thick_pure_bquadrupole(tm, kc, p_over_q_ref, bunch, bm2, L)                                     = error("Undefined for tracking method $tm")
@inline thick_bquadrupole(tm, kc, p_over_q_ref, bunch, bmultipoleparams, L)                             = error("Undefined for tracking method $tm")
@inline thick_pure_bmultipole(tm, kc, p_over_q_ref, bunch, bmk, L)                                      = error("Undefined for tracking method $tm")
@inline thick_bmultipole(tm, kc, p_over_q_ref, bunch, bmultipoleparams, L)                              = error("Undefined for tracking method $tm")
@inline thick_bmultipole_rf(tm, kc, p_over_q_ref, bunch, bmultipoleparams, rfparams, beamlineparams, L) = error("Undefined for tracking method $tm")

# === Elements with curving coordinate system "bend" === #
# "Bend" means ONLY a coordinate system curvature through the element.
# It does NOT IMPLY ANY DIPOLE FIELD! Bend specifies if the integration 
# path is curving but does not IMPACT THE PHYSICS inside.
# "Pure" means only ONE SINGLE MULTIPOLE
# When "pure" is not present, it means that at least one higher order 
# multipole exists.

# SciBmad will probably not support thin bends ever but I leave them here for now
@inline thin_bend_no_field(tm, kc, p_over_q_ref, bunch, bendparams)                      = error("Undefined for tracking method $tm")
@inline thin_bend_pure_bsolenoid(tm, kc, p_over_q_ref, bunch, bendparams, bm0)           = error("Undefined for tracking method $tm")
@inline thin_bend_bsolenoid(tm, kc, p_over_q_ref, bunch, bendparams, bmultipoleparams)   = error("Undefined for tracking method $tm")
@inline thin_bend_pure_bdipole(tm, kc, p_over_q_ref, bunch, bendparams, bm1)             = error("Undefined for tracking method $tm")
@inline thin_bend_bdipole(tm, kc, p_over_q_ref, bunch, bendparams, bmultipoleparams)     = error("Undefined for tracking method $tm")
@inline thin_bend_pure_bquadrupole(tm, kc, p_over_q_ref, bunch, bendparams, bm2)         = error("Undefined for tracking method $tm")
@inline thin_bend_bquadrupole(tm, kc, p_over_q_ref, bunch, bendparams, bmultipoleparams) = error("Undefined for tracking method $tm")
@inline thin_bend_pure_bmultipole(tm, kc, p_over_q_ref, bunch, bendparams, bmk)          = error("Undefined for tracking method $tm")
@inline thin_bend_bmultipole(tm, kc, p_over_q_ref, bunch, bendparams, bmultipoleparams)  = error("Undefined for tracking method $tm")

@inline thick_bend_no_field(tm, kc, p_over_q_ref, bunch, bendparams, L)                      = error("Undefined for tracking method $tm")
@inline thick_bend_pure_bsolenoid(tm, kc, p_over_q_ref, bunch, bendparams, bm0, L)           = error("Undefined for tracking method $tm")
@inline thick_bend_bsolenoid(tm, kc, p_over_q_ref, bunch, bendparams, bmultipoleparams, L)   = error("Undefined for tracking method $tm")
@inline thick_bend_pure_bdipole(tm, kc, p_over_q_ref, bunch, bendparams, bm1, L)             = error("Undefined for tracking method $tm")
@inline thick_bend_bdipole(tm, kc, p_over_q_ref, bunch, bendparams, bmultipoleparams, L)     = error("Undefined for tracking method $tm")
@inline thick_bend_pure_bquadrupole(tm, kc, p_over_q_ref, bunch, bendparams, bm2, L)         = error("Undefined for tracking method $tm")
@inline thick_bend_bquadrupole(tm, kc, p_over_q_ref, bunch, bendparams, bmultipoleparams, L) = error("Undefined for tracking method $tm")
@inline thick_bend_pure_bmultipole(tm, kc, p_over_q_ref, bunch, bendparams, bmk, L)          = error("Undefined for tracking method $tm")
@inline thick_bend_bmultipole(tm, kc, p_over_q_ref, bunch, bendparams, bmultipoleparams, L)  = error("Undefined for tracking method $tm")


# === Elements thin vs thick check === #
@inline pure_rf(tm, kc, p_over_q_ref, bunch, rfparams, beamlineparams, L)                          = L == 0 ? thin_pure_rf(tm, kc, p_over_q_ref, bunch, rfparams, beamlineparams)                         : thick_pure_rf(tm, kc, p_over_q_ref, bunch, rfparams, beamlineparams, L)
@inline pure_bsolenoid(tm, kc, p_over_q_ref, bunch, bm0, L)                                   = L == 0 ? thin_pure_bsolenoid(tm, kc, p_over_q_ref, bunch, bm0)                                  : thick_pure_bsolenoid(tm, kc, p_over_q_ref, bunch, bm0, L)      
@inline bsolenoid(tm, kc, p_over_q_ref, bunch, bmultipoleparams, L)                           = L == 0 ? thin_bsolenoid(tm, kc, p_over_q_ref, bunch, bmultipoleparams)                          : thick_bsolenoid(tm, kc, p_over_q_ref, bunch, bmultipoleparams, L)       
@inline pure_bdipole(tm, kc, p_over_q_ref, bunch, bm1, L)                                     = L == 0 ? thin_pure_bdipole(tm, kc, p_over_q_ref, bunch, bm1)                                    : thick_pure_bdipole(tm, kc, p_over_q_ref, bunch, bm1, L)          
@inline bdipole(tm, kc, p_over_q_ref, bunch, bmultipoleparams, L)                             = L == 0 ? thin_bdipole(tm, kc, p_over_q_ref, bunch, bmultipoleparams)                            : thick_bdipole(tm, kc, p_over_q_ref, bunch, bmultipoleparams, L)             
@inline pure_bquadrupole(tm, kc, p_over_q_ref, bunch, bm2, L)                                 = L == 0 ? thin_pure_bquadrupole(tm, kc, p_over_q_ref, bunch, bm2)                                : thick_pure_bquadrupole(tm, kc, p_over_q_ref, bunch, bm2, L)        
@inline bquadrupole(tm, kc, p_over_q_ref, bunch, bmultipoleparams, L)                         = L == 0 ? thin_bquadrupole(tm, kc, p_over_q_ref, bunch, bmultipoleparams)                        : thick_bquadrupole(tm, kc, p_over_q_ref, bunch, bmultipoleparams, L)           
@inline pure_bmultipole(tm, kc, p_over_q_ref, bunch, bmk, L)                                  = L == 0 ? thin_pure_bmultipole(tm, kc, p_over_q_ref, bunch, bmk)                                 : thick_pure_bmultipole(tm, kc, p_over_q_ref, bunch, bmk, L)                   
@inline bmultipole(tm, kc, p_over_q_ref, bunch, bmultipoleparams, L)                          = L == 0 ? thin_bmultipole(tm, kc, p_over_q_ref, bunch, bmultipoleparams)                         : thick_bmultipole(tm, kc, p_over_q_ref, bunch, bmultipoleparams, L)
@inline bmultipole_rf(tm, kc, p_over_q_ref, bunch, bmultipoleparams, rfparams, beamlineparams, L)  = L == 0 ? thin_bmultipole_rf(tm, kc, p_over_q_ref, bunch, bmultipoleparams, rfparams, beamlineparams) : thick_bmultipole_rf(tm, kc, p_over_q_ref, bunch, bmultipoleparams, rfparams, beamlineparams, L)        
@inline bend_no_field(tm, kc, p_over_q_ref, bunch, bendparams, L)                             = L == 0 ? thin_bend_no_field(tm, kc, p_over_q_ref, bunch, bendparams)                            : thick_bend_no_field(tm, kc, p_over_q_ref, bunch, bendparams, L)
@inline bend_pure_bsolenoid(tm, kc, p_over_q_ref, bunch, bendparams, bm0, L)                  = L == 0 ? thin_bend_pure_bsolenoid(tm, kc, p_over_q_ref, bunch, bendparams, bm0)                 : thick_bend_pure_bsolenoid(tm, kc, p_over_q_ref, bunch, bendparams, bm0, L)      
@inline bend_bsolenoid(tm, kc, p_over_q_ref, bunch, bendparams, bmultipoleparams, L)          = L == 0 ? thin_bend_bsolenoid(tm, kc, p_over_q_ref, bunch, bendparams, bmultipoleparams)         : thick_bend_bsolenoid(tm, kc, p_over_q_ref, bunch, bendparams, bmultipoleparams, L)         
@inline bend_pure_bdipole(tm, kc, p_over_q_ref, bunch, bendparams, bm1, L)                    = L == 0 ? thin_bend_pure_bdipole(tm, kc, p_over_q_ref, bunch, bendparams, bm1)                   : thick_bend_pure_bdipole(tm, kc, p_over_q_ref, bunch, bendparams, bm1, L)          
@inline bend_bdipole(tm, kc, p_over_q_ref, bunch, bendparams, bmultipoleparams, L)            = L == 0 ? thin_bend_bdipole(tm, kc, p_over_q_ref, bunch, bendparams, bmultipoleparams)           : thick_bend_bdipole(tm, kc, p_over_q_ref, bunch, bendparams, bmultipoleparams, L)             
@inline bend_pure_bquadrupole(tm, kc, p_over_q_ref, bunch, bendparams, bm2, L)                = L == 0 ? thin_bend_pure_bquadrupole(tm, kc, p_over_q_ref, bunch, bendparams, bm2)               : thick_bend_pure_bquadrupole(tm, kc, p_over_q_ref, bunch, bendparams, bm2, L)        
@inline bend_bquadrupole(tm, kc, p_over_q_ref, bunch, bendparams, bmultipoleparams, L)        = L == 0 ? thin_bend_bquadrupole(tm, kc, p_over_q_ref, bunch, bendparams, bmultipoleparams)       : thick_bend_bquadrupole(tm, kc, p_over_q_ref, bunch, bendparams, bmultipoleparams, L)           
@inline bend_pure_bmultipole(tm, kc, p_over_q_ref, bunch, bendparams, bmk, L)                 = L == 0 ? thin_bend_pure_bmultipole(tm, kc, p_over_q_ref, bunch, bendparams, bmk)                : thick_bend_pure_bmultipole(tm, kc, p_over_q_ref, bunch, bendparams, bmk, L)                   
@inline bend_bmultipole(tm, kc, p_over_q_ref, bunch, bendparams, bmultipoleparams, L)         = L == 0 ? thin_bend_bmultipole(tm, kc, p_over_q_ref, bunch, bendparams, bmultipoleparams)        : thick_bend_bmultipole(tm, kc, p_over_q_ref, bunch, bendparams, bmultipoleparams, L)                      
