// rw07-duration-intfp32-v3: private closed-duration arithmetic, not a generic runtime.
// All helpers take finite operands; positive FMA additionally requires nonnegative
// operands. Duration validation and the fixed Sigmoid coefficients establish these
// premises. No floating arithmetic is used to implement rounding. vec2<u32> stores
// low/high words; normalization retains every 48-bit product bit before alignment.
#pragma once
namespace rw07 {
inline constexpr const char* DurationExactFp32 = R"WGSL(
struct RW07D { mant:u32, exponent:i32, }
fn rw07_decode(bits:u32)->RW07D {
 let fraction=bits&0x7fffffu;let field=(bits>>23u)&255u;
 if(field!=0u){return RW07D(fraction|0x800000u,i32(field)-127);}
 if(fraction==0u){return RW07D(0u,0);}
 let shift=countLeadingZeros(fraction)-8u;
 return RW07D(fraction<<shift,-126-i32(shift));
}
fn rw07_nz(a:vec2<u32>)->bool{return (a.x|a.y)!=0u;}
fn rw07_less(a:vec2<u32>,b:vec2<u32>)->bool{return a.y<b.y||(a.y==b.y&&a.x<b.x);}
fn rw07_uadd(a:vec2<u32>,b:vec2<u32>)->vec2<u32>{let low=a.x+b.x;return vec2<u32>(low,a.y+b.y+select(0u,1u,low<a.x));}
fn rw07_usub(a:vec2<u32>,b:vec2<u32>)->vec2<u32>{return vec2<u32>(a.x-b.x,a.y-b.y-select(0u,1u,a.x<b.x));}
fn rw07_shl(a:vec2<u32>,shift:u32)->vec2<u32>{
 if(shift==0u){return a;}if(shift>=64u){return vec2<u32>(0u);}
 if(shift>=32u){return vec2<u32>(0u,a.x<<(shift-32u));}
 return vec2<u32>(a.x<<shift,(a.y<<shift)|(a.x>>(32u-shift)));
}
fn rw07_shr(a:vec2<u32>,shift:u32)->vec2<u32>{
 if(shift==0u){return a;}if(shift>=64u){return vec2<u32>(0u);}
 if(shift>=32u){return vec2<u32>(a.y>>(shift-32u),0u);}
 return vec2<u32>((a.x>>shift)|(a.y<<(32u-shift)),a.y>>shift);
}
fn rw07_low_nonzero(a:vec2<u32>,count:u32)->bool{
 if(count==0u){return false;}if(count>=64u){return rw07_nz(a);}
 if(count==32u){return a.x!=0u;}
 if(count<32u){return (a.x&((1u<<count)-1u))!=0u;}
 return a.x!=0u||(a.y&((1u<<(count-32u))-1u))!=0u;
}
fn rw07_jam(a:vec2<u32>,shift:u32)->vec2<u32>{var n=rw07_shr(a,shift);n.x=n.x|select(0u,1u,rw07_low_nonzero(a,shift));return n;}
fn rw07_high(a:vec2<u32>)->u32{if(a.y!=0u){return 63u-countLeadingZeros(a.y);}return 31u-countLeadingZeros(a.x);}
fn rw07_mul24(a:u32,b:u32)->vec2<u32>{
 let al=a&65535u;let ah=a>>16u;let bl=b&65535u;let bh=b>>16u;
 let p1=al*bh;let p2=ah*bl;
 var n=vec2<u32>(al*bl,ah*bh);
 n=rw07_uadd(n,vec2<u32>(p1<<16u,p1>>16u));
 return rw07_uadd(n,vec2<u32>(p2<<16u,p2>>16u));
}
fn rw07_pack(raw:vec2<u32>,original_exponent:i32)->u32{
 if(!rw07_nz(raw)){return 0u;}let high=rw07_high(raw);var n=raw;
 if(high>60u){n=rw07_jam(n,high-60u);}else{n=rw07_shl(n,60u-high);}
 var exponent=original_exponent+i32(high)-60;
 let shift=37u+u32(max(0,-126-exponent));
 var quotient=rw07_shr(n,shift).x;
 let guard=(rw07_shr(n,shift-1u).x&1u)!=0u;
 let sticky=rw07_low_nonzero(n,shift-1u);
 if(guard&&(sticky||(quotient&1u)!=0u)){quotient+=1u;}
 if(exponent< -126){return quotient;}
 if(quotient>=0x1000000u){quotient=quotient>>1u;exponent+=1;}
 if(exponent>127){return 0x7f800000u;}
 return (u32(exponent+127)<<23u)|(quotient&0x7fffffu);
}
fn rw07_fma_positive_bits(a:u32,b:u32,c:u32)->u32{
 let da=rw07_decode(a);let db=rw07_decode(b);let dc=rw07_decode(c);
 if(da.mant==0u||db.mant==0u){return c;}
 let product=rw07_mul24(da.mant,db.mant);let high=rw07_high(product);
 let product_exponent=da.exponent+db.exponent-46+i32(high);
 let n=rw07_shl(product,60u-high);
 if(dc.mant==0u){return rw07_pack(n,product_exponent);}
 let z=rw07_shl(vec2<u32>(dc.mant,0u),37u);
 let exponent=max(product_exponent,dc.exponent);
 return rw07_pack(rw07_uadd(rw07_jam(n,u32(exponent-product_exponent)),rw07_jam(z,u32(exponent-dc.exponent))),exponent);
}
fn rw07_fma_positive(a:f32,b:f32,c:f32)->f32{return bitcast<f32>(rw07_fma_positive_bits(bitcast<u32>(a),bitcast<u32>(b),bitcast<u32>(c)));}
fn rw07_mul(a:f32,b:f32)->f32{
 let ab=bitcast<u32>(a);let bb=bitcast<u32>(b);
 return bitcast<f32>(((ab^bb)&0x80000000u)|rw07_fma_positive_bits(ab&0x7fffffffu,bb&0x7fffffffu,0u));
}
fn rw07_add(a:f32,b:f32)->f32{
 let ab=bitcast<u32>(a);let bb=bitcast<u32>(b);let da=rw07_decode(ab);let db=rw07_decode(bb);
 let sa=ab&0x80000000u;let sb=bb&0x80000000u;
 if(da.mant==0u&&db.mant==0u){return bitcast<f32>(sa&sb);}
 if(da.mant==0u){return b;}if(db.mant==0u){return a;}
 let exponent=max(da.exponent,db.exponent);
 let na=rw07_jam(rw07_shl(vec2<u32>(da.mant,0u),37u),u32(exponent-da.exponent));
 let nb=rw07_jam(rw07_shl(vec2<u32>(db.mant,0u),37u),u32(exponent-db.exponent));
 if(sa==sb){return bitcast<f32>(sa|rw07_pack(rw07_uadd(na,nb),exponent));}
 if(all(na==nb)){return 0.0;}
 if(rw07_less(na,nb)){return bitcast<f32>(sb|rw07_pack(rw07_usub(nb,na),exponent));}
 return bitcast<f32>(sa|rw07_pack(rw07_usub(na,nb),exponent));
}
fn rw07_add4(a:vec4<f32>,b:vec4<f32>)->vec4<f32>{return vec4<f32>(rw07_add(a.x,b.x),rw07_add(a.y,b.y),rw07_add(a.z,b.z),rw07_add(a.w,b.w));}
fn rw07_div(a:f32,b:f32)->f32{
 let ab=bitcast<u32>(a);let bb=bitcast<u32>(b);let da=rw07_decode(ab);let db=rw07_decode(bb);
 let sign=(ab^bb)&0x80000000u;if(db.mant==0u){return bitcast<f32>(sign|0x7fc00000u);}
 if(da.mant==0u){return bitcast<f32>(sign);}
 var exponent=da.exponent-db.exponent;var remainder=da.mant;
 if(remainder<db.mant){remainder=remainder<<1u;exponent-=1;}
 remainder-=db.mant;var quotient=1u;
 for(var i=0u;i<23u;i++){remainder=remainder<<1u;quotient=quotient<<1u;if(remainder>=db.mant){quotient=quotient|1u;remainder-=db.mant;}}
 if(exponent< -126){
  let shift=u32(-126-exponent);if(shift>24u){return bitcast<f32>(sign);}
  let guard=((quotient>>(shift-1u))&1u)!=0u;
  let sticky=(quotient&((1u<<(shift-1u))-1u))!=0u||remainder!=0u;
  quotient=quotient>>shift;if(guard&&(sticky||(quotient&1u)!=0u)){quotient+=1u;}
  return bitcast<f32>(sign|quotient);
 }
 if(remainder*2u>db.mant||(remainder*2u==db.mant&&(quotient&1u)!=0u)){quotient+=1u;}
 if(quotient>=0x1000000u){quotient=quotient>>1u;exponent+=1;}
 if(exponent>127){return bitcast<f32>(sign|0x7f800000u);}
 return bitcast<f32>(sign|(u32(exponent+127)<<23u)|(quotient&0x7fffffu));
}
)WGSL";
}
