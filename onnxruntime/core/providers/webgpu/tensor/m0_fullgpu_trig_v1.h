// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <cstdint>

namespace onnxruntime::webgpu::m0_trig_detail {
// Conservative source-level product count: both16-digit2/pi integer products,
// three fractional reconstruction products and14 sine/cosine polynomial products.
// This is a scheduling envelope, not measured hardware instructions or FLOPs.
inline constexpr uint32_t kSincosProductsPerElementV1=2*16+3+14;
// WGSL only bounds builtin sin/cos accuracy on [-pi, pi]. Original phase
// accumulation and Snake activations are not restricted to that interval.
// Use an exact-product split pi/2 reduction for bounded arguments and an
// integer 256-bit 2/pi product for large arguments. Float modulo by a
// rounded pi would lose the phase of large arguments.
inline constexpr const char* Trig = R"WGSL(
fn rw07_trig_f32(bits:u32)->f32 {return bitcast<f32>(bits);}
fn rw07_trig_bit(p:array<u32,18>,at:u32)->u32 {
 if(at>=288u){return 0u;}return (p[at/16u]>>(at%16u))&1u;
}
fn rw07_trig_word(p:array<u32,18>,at:u32)->u32 {
 let i=at/16u;let b=at%16u;
 if(b==0u){return p[i]|(p[i+1u]<<16u);}
 return (p[i]>>b)|(p[i+1u]<<(16u-b))|(p[i+2u]<<(32u-b));
}
fn rw07_trig_reduce(magnitude:u32)->vec2<f32> {
 // Little-endian base-2^16 digits of floor((2/pi)*2^256).
 let digits=array<u32,16>(0xc561u,0xdebbu,0x63abu,0xfe51u,0x9041u,0x3c43u,0x9599u,0xdb62u,
                         0xddc0u,0xf534u,0x57d1u,0xfc27u,0x1529u,0x4e44u,0x836eu,0xa2f9u);
 let mantissa=(magnitude&0x7fffffu)|0x800000u;
 let low=mantissa&65535u;let high=mantissa>>16u;
 var p:array<u32,18>;var carry=0u;
 for(var i=0u;i<16u;i++){
  let product=digits[i]*low+carry;p[i]=product&65535u;carry=product>>16u;
 }
 p[16]=carry;carry=0u;
 for(var i=0u;i<16u;i++){
  let product=digits[i]*high+p[i+1u]+carry;p[i+1u]=product&65535u;carry=product>>16u;
 }
 p[17]=carry;
 // x = mantissa*2^(exponent-23), hence x*2/pi = p/2^(279-exponent).
 // The caller only reaches this path for |x| > pi/4, so shift is in [152,280].
 let exponent=i32((magnitude>>23u)&255u)-127;let shift=u32(279-exponent);
 let round_up=rw07_trig_bit(p,shift-1u);
 let quadrant=(rw07_trig_bit(p,shift)+2u*rw07_trig_bit(p,shift+1u)+round_up)&3u;
 var hi=rw07_trig_word(p,shift-32u);var lo=rw07_trig_word(p,shift-64u);
 if(round_up!=0u){lo=0u-lo;hi=(~hi)+select(0u,1u,lo==0u);}
 let fraction=f32(hi)*0x1p-32f+f32(lo)*0x1p-64f;
 let reduced=fraction*1.57079632679489661923;
 return vec2<f32>(select(reduced,-reduced,round_up!=0u),f32(quadrant));
}
fn rw07_sincos(x:f32)->vec2<f32> {
 let bits=bitcast<u32>(x);let magnitude=bits&0x7fffffffu;
 if(magnitude>=0x7f800000u){let nan=rw07_trig_f32(0x7fc00000u);return vec2<f32>(nan);}
 if(magnitude<0x39800000u){return vec2<f32>(x,1.0);}
 var r=bitcast<f32>(magnitude);var quadrant=0u;
 if(magnitude>0x3f490fdbu){
  if(magnitude<=0x46000000u){
   // For |x|<=8192, q<=5215. These first two products are exact:
   // 1.5703125=201/128 and 0.0004837512969970703125=2029/2^22.
   // Subtract the split pi/2 terms before the small rounded tail.
   let q=round(r*0.63661977236758134308);
   r=((r-q*1.5703125)-q*0.0004837512969970703125)-q*7.549789954891882e-8;
   quadrant=u32(q)&3u;
  }else{
   let reduced=rw07_trig_reduce(magnitude);r=reduced.x;quadrant=u32(reduced.y);
  }
 }
 // Taylor polynomials on [-pi/4,pi/4]: truncation <4e-13 for cos,
 // <3e-14 for sin; FP32 evaluation error is tested separately.
 let z=r*r;
 var sp=1.0/6227020800.0;
 sp=sp*z-1.0/39916800.0;sp=sp*z+1.0/362880.0;
 sp=sp*z-1.0/5040.0;sp=sp*z+1.0/120.0;sp=sp*z-1.0/6.0;
 let sine=r+r*z*sp;
 var cp=1.0/479001600.0;
 cp=cp*z-1.0/3628800.0;cp=cp*z+1.0/40320.0;
 cp=cp*z-1.0/720.0;cp=cp*z+1.0/24.0;cp=cp*z-0.5;
 let cosine=1.0+z*cp;
 var result=vec2<f32>(sine,cosine);
 switch quadrant {
  case 1u:{result=vec2<f32>(cosine,-sine);}
  case 2u:{result=vec2<f32>(-sine,-cosine);}
  case 3u:{result=vec2<f32>(-cosine,sine);}
  default:{}
 }
 if((bits&0x80000000u)!=0u){result.x=-result.x;}
 return result;
}
fn rw07_sin(x:f32)->f32 {return rw07_sincos(x).x;}
fn rw07_cos(x:f32)->f32 {return rw07_sincos(x).y;}
)WGSL";

// WGSL's builtin atan permits4096ULP: it cannot guarantee the original
// absolute1e-4 operator tolerance. Reduce to |r|<=tan(pi/8), then evaluate
// the alternating Taylor polynomial through r^19. Its truncation bound is
// tan(pi/8)^21/21; FP32 evaluation is checked by the numerical test corpus.
// https://www.w3.org/TR/WGSL/#floating-point-accuracy
inline constexpr const char* Atan = R"WGSL(
// A runtime bitcast permits nonfinite results; a WGSL constant expression does not.
fn rw07_atan_f32(bits:u32)->f32 {return bitcast<f32>(bits);}
fn rw07_atan(x:f32)->f32 {
 let bits=bitcast<u32>(x);let magnitude=bits&0x7fffffffu;let sign=bits&0x80000000u;
 if(magnitude>0x7f800000u){return rw07_atan_f32(0x7fc00000u);}
 if(magnitude==0x7f800000u){return bitcast<f32>(sign|0x3fc90fdbu);}
 // For |x|<2^-12, |atan(x)-x|<|x|^3/3. Preserve zeros/subnormals bitwise.
 if(magnitude<0x39800000u){return x;}
 // Rounded FLOAT values of sqrt(2)-1 and sqrt(2)+1, in dimensionless radians.
 let tan_pi_over_eight=bitcast<f32>(0x3ed413cdu);
 let tan_three_pi_over_eight=bitcast<f32>(0x401a827au);
 let a=bitcast<f32>(magnitude);var r=a;var offset=0.0;
 if(a>tan_three_pi_over_eight){
  r=-1.0/a;offset=bitcast<f32>(0x3fc90fdbu);
 }else if(a>tan_pi_over_eight){
  r=(a-1.0)/(a+1.0);offset=bitcast<f32>(0x3f490fdbu);
 }
 let z=r*r;var p=-1.0/19.0;
 p=p*z+1.0/17.0;p=p*z-1.0/15.0;p=p*z+1.0/13.0;
 p=p*z-1.0/11.0;p=p*z+1.0/9.0;p=p*z-1.0/7.0;
 p=p*z+1.0/5.0;p=p*z-1.0/3.0;
 let reduced=r+(r*z)*p;let result=offset+reduced;
 return bitcast<f32>(bitcast<u32>(result)|sign);
}
)WGSL";
}  // namespace onnxruntime::webgpu::m0_trig_detail
