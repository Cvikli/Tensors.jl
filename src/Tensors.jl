module Tensors



using CUDA
using CUDA: i32
using Pythonish


@inline Mi_Bi_iter(I, row_len) = begin 
  s = cld(I, row_len)
  I-(s-1)*row_len, s 
end


tensor_assign_MBTN_HXTN(a, b, h_slice, v_slice) = @inbounds begin
	_,_,T, N= size(a)
	i=1
	@fastmath @simd for ni in 1:N
		for ti in 1:T
			for vi in v_slice
				for hi in h_slice
					a[i] = b[hi,vi,ti,ni]
					i+=1
				end
			end
		end
	end
end
tensor_assign_MBN_HX1N(a, b, h_slice, v_slice) = @inbounds begin
	N = size(a,4)
	@assert N == size(b,4)
	size(b,4) == 0 && return
	i=1
	@fastmath @simd for n in 1:N
		for vi in v_slice
			for hi in h_slice
				a[i] = b[hi,vi,1,n]
				i += 1
			end
		end
	end
end
tensor_assign_M_H(a, b, h_slice, v_slice) = a .= b[h_slice]


# CUDA_BTX_assign!()

tensor_assign_B1N_X1N(a, b, v_slice) = @inbounds begin
	B,T,N = size(a)
	i=1
	@simd for ni in 1:N
		for bi in v_slice
			a[i] = b[bi,1,ni]
			i+=1
		end
	end
end
tensor_assign_B1I_X1I(a, b, v_slice, idxs) = @inbounds begin
	i=1
	@simd for ni in idxs
		for bi in v_slice
			a[i] = b[bi,1,ni]
			i+=1
		end
	end
end
tensor_assign_BTN_XTN(a, b, v_slice) = @inbounds begin
	i=1
	B, T, N = size(a)
	@simd for ni in 1:N
		for ti in 1:T
			for bi in v_slice
				a[i] = b[bi,ti,ni]
				i+=1
			end
		end
	end
end
tensor_assign_BTN_XTN(a, b, idxs, T, N) = @inbounds begin
	i=1
	@simd for ni in 1:N
		for ti in 1:T
			for bi in idxs
				a[i] = b[bi,ti,ni]
				i+=1
			end
		end
	end
end
tensor_assigns_BTN_BTX!(a, b, B, T, idxs) = begin
	nth = 512
	BT  = B*T
	BTX = B*T*len(idxs)
	@sync @cuda threads=nth blocks=cld(BTX, nth) CUDA_assign_BTN_BTX!(a, b, idxs, BT, BTX)
end

tensor_assigns_BTX_BTN!(a, b, idxs) = begin
	B, T, N = size(a)
	nth = 512
	BT  = B*T
	BTX = B*T*len(idxs)
	@sync @cuda threads=nth blocks=cld(BTX, nth) CUDA_assign_BTX_BTN!(a, b, idxs, BT, BTX)
end
tensor_assigns_BTX_BTN!(a, b, idxs, B, T) = begin
	nth = 512
	BT  = B*T
	BTX = B*T*len(idxs)
	@sync @cuda threads=nth blocks=cld(BTX, nth) CUDA_assign_BTX_BTN!(a, b, idxs, BT, BTX)
end
tensor_N1X_assigns!(a, b, idxs, B, T) = begin
	nth = 512
	BT  = B*T
	B1X = B*len(idxs)
	@sync @cuda threads=nth blocks=cld(B1X, nth) CUDA_assign_B1X_BTN!(a, b, idxs, B, BT, B1X)
end

CUDA_assign_BTN_BTX!(a, b, idxs, BT, BTX) = @inbounds begin
	I = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
	I > BTX && return 
  off, Pi  = (I - 1i32) % BT + 1i32, cld(I, BT)
	cidxs = CUDA.Const(idxs)
	Voff, Poff = (Pi - 1i32) * BT, (cidxs[Pi] - 1i32) * BT  
	a[Voff + off] = b[Poff + off]
	return
end
CUDA_assign_BTX_BTN!(a, b, idxs, BT, BTX) = @inbounds begin
	I = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
	I > BTX && return 
  off, Pi  = (I - 1i32) % BT + 1i32, cld(I, BT)
	cidxs = CUDA.Const(idxs)
	Poff, Voff = (cidxs[Pi] - 1i32) * BT, (Pi - 1i32) * BT  
	a[Poff + off] = b[Voff + off]
	return
end

CUDA_assign_B1X_BTN!(a, b, idxs, B, BT, B1X) = @inbounds begin
	I = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
	I > B1X && return 
  off, Pi  = (I - 1i32) % B + 1i32, cld(I, B)
	cidxs = CUDA.Const(idxs)
	Poff, Voff = (cidxs[Pi] - 1i32) * BT, (Pi - 1i32) * B  
	a[Poff + off] = b[Voff + off]
	return
end

tensor_sub!(a, b, idxs, B, T) = begin
	nth = 512
	BT  = B*T
	X = len(idxs)
	@sync @cuda threads=nth blocks=cld(X, nth) CUDA_BTX_11X_sub!(a, b, idxs, BT, X)
end
tensor_N_sub!(a, b, idxs, B, T) = begin
	nth = 512
	T   = T
	BT  = B*T
	X   = B*len(idxs)
	@sync @cuda threads=nth blocks=cld(X, nth) CUDA_BTX_B1X_sub!(a, b, idxs, B, T, BT, X)
end
tensor_N1X_sub!(a, b, idxs, B, T) = begin
	nth = 512
	BT  = B*T
	B1X = B*len(idxs)
	@sync @cuda threads=nth blocks=cld(B1X, nth) CUDA_B1X_sub!(a, b, idxs, B, BT, B1X)
end

CUDA_BTX_11X_sub!(a, b, idxs, BT, X) = @inbounds begin
	I = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
	I > X && return 
	Pi=I
	cidxs = CUDA.Const(idxs)
	Poff = (cidxs[Pi] - 1i32) * BT  
	v = b[Pi]
	for i in 1:BT
		a[Poff + i] -= v
	end
	return
end
CUDA_BTX_B1X_sub!(a, b, idxs, B, T, BT, X) = @inbounds begin
	I = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
	I > X && return 
	Boff, Pi = Mi_Bi_iter(I, B) 
	cidxs = CUDA.Const(idxs)
	Poff = (cidxs[Pi] - 1i32) * BT  
	Off = Poff + Boff
	v = b[Off]
	for i in 0:T-1
		a[Off + i] -= v
	end
	return
end
CUDA_B1X_sub!(a, b, idxs, B, BT, B1X) = @inbounds begin
	I = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
	I > B1X && return 
  off, Pi  = (I - 1i32) % B + 1i32, cld(I, B)
	cidxs = CUDA.Const(idxs)
	Poff, Voff = (cidxs[Pi] - 1i32) * BT, (Pi - 1i32) * B  
	a[Poff + off] -= b[Voff + off]
	return
end



tensor_assigns_mP_m11p!(a, b, ⅀M, Xidx, M, B, T, idxs) = begin
	@assert false " nincs használva Xidx!! LOL!"
	nth = 512
	MBT = M*B*T
	MBTP = 1*1*1*length(idxs)
	@sync @cuda threads=nth blocks=cld(MBTP, nth) CUDA_assign_mP_m11p!(a, b, ⅀M, idxs, MBT, MBTP)
end
CUDA_assign_mP_m11p!(a, b, ⅀M, idxs, MBT, MBTP) = @inbounds begin
	I = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
	I > MBTP && return 
  Pi    = I
	cidxs = CUDA.Const(idxs)
	Voff, Poff = (Pi - 1i32) * ⅀M, (cidxs[Pi] - 1i32) * MBT  
	a[Voff+1] = b[Poff+1]
	return
end

tensor_assigns_mBTp_mP!(a, b, ⅀M, Xidx, M, B, T, idxs) = begin
	@assert false " nincs használva Xidx!! LOL!"
	nth = 512
	BT   = B*T
	MBT  = M*B*T
	MBTP = 1*B*T*length(idxs)
	@sync @cuda threads=nth blocks=cld(MBTP, nth) CUDA_assign_mBTp_mP!(a, b, ⅀M, idxs, BT, MBT, MBTP)
end
CUDA_assign_mBTp_mP!(a, b, ⅀M, idxs, BT, MBT, MBTP) = @inbounds begin
	I = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
	I > MBTP && return 
  off, Pi  = (I - 1i32) % BT + 1i32, cld(I, BT)
	cidxs = CUDA.Const(idxs)
	Voff, Poff = (cidxs[Pi] - 1i32) * MBT, (Pi - 1i32) * ⅀M
	a[Voff+off] = b[Poff+1]
	return
end
assigns_mP_m11p!(a, b, Xidx, ⅀M, Mi, MBT, idxs) = begin
	for Pi in 1:length(idxs)
		Voff = (Pi - 1i32) * ⅀M
		Poff = (idxs[Pi] - 1i32) * MBT  
		a[Voff+Xidx] = b[Poff+Mi]
	end
end
assign_mBTp_mP!(a, b, Xidx, ⅀M, Mi, M, BT, MBT, idxs) = begin
	for Pi in 1:length(idxs)
		Voff = (idxs[Pi] - 1i32) * MBT + Mi
		Poff = (Pi - 1i32) * ⅀M        + Xidx
		bPoff = b[Poff]
		for off in 0:BT-1
			a[Voff+off * M] = bPoff
		end
	end
end


rev_bc_sum(res, arr, idxs, B,T) = begin
	I = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  I > len(idxs) && return 
	cg = CUDA.Const(arr)
  idx = CUDA.Const(idxs)[I]
  v_sum = 0f0
	poff = (idx-1) * B * T + 1
	poff_end = poff + B * T
	while poff < poff_end
		v_sum += cg[poff]
		poff+=1; end 
  res[I] = v_sum
	return
end
sum_bc_grad!(revbc_grad, g, idxs) = begin
  B, T, = size(g)
	CUDA.@sync @cuda threads=128 blocks=cld(len(idxs),128) rev_bc_sum(revbc_grad, g, idxs,B,T)
end
subtract_on_idxs(arr, idxs, arr2, B,T) = @inbounds begin
	I = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  I > B*T && return 
  # I > len(idxs) && return 
  cidxs = CUDA.Const(idxs)
	for i in 1:len(cidxs)
		idx = cidxs[i]
    arr[I,1,idx] -= arr2[i]
  end
  return
end

end # module Tensors
